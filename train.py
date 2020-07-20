import os
import re
import torch
import random
import time
import logging
import argparse
import subprocess
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import (BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup)

from modeling import RepBERT_Train
from dataset import MSMARCODataset, get_collate_function
from utils import generate_rank, eval_results

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_model(model, output_dir, save_name, args):
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(save_dir)
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))


def train(args, model):
    """ Train the model """
    tb_writer = SummaryWriter(args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = MSMARCODataset("train", args.msmarco_dir, 
            args.collection_memmap_dir, args.tokenize_dir,
            args.max_query_length, args.max_doc_length)

    # NOTE: Must Sequential! Pos, Neg, Pos, Neg, ...
    train_sampler = SequentialSampler(train_dataset) 
    collate_fn = get_collate_function(mode="train")
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=args.train_batch_size, num_workers=args.data_num_workers, 
        collate_fn=collate_fn)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, (batch, _, _) in enumerate(epoch_iterator):

            batch = {k:v.to(args.device) for k, v in batch.items()}
            model.train()            
            outputs = model(**batch)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.evaluate_during_training and (global_step % args.training_eval_steps == 0):
                    mrr = evaluate(args, model, mode="dev", prefix="step_{}".format(global_step))
                    tb_writer.add_scalar('dev/MRR@10', mrr, global_step)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    cur_loss =  (tr_loss - logging_loss)/args.logging_steps
                    tb_writer.add_scalar('train/loss', cur_loss, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    save_model(model, args.model_save_dir, 'ckpt-{}'.format(global_step), args)


def evaluate(args, model, mode, prefix):
    eval_output_dir = args.eval_save_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
  
    eval_dataset = MSMARCODataset(mode, args.msmarco_dir, 
            args.collection_memmap_dir, args.tokenize_dir,
            args.max_query_length, args.max_doc_length)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    collate_fn = get_collate_function(mode=mode)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size,
        num_workers=args.data_num_workers, collate_fn=collate_fn)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    output_file_path = f"{eval_output_dir}/{prefix}.{mode}.score.tsv"
    with open(output_file_path, 'w') as outputfile:
        for batch, qids, docids in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                batch = {k:v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                scores = torch.diagonal(outputs[0]).detach().cpu().numpy()
                assert len(qids) == len(docids) == len(scores)
                for qid, docid, score in zip(qids, docids, scores):
                    outputfile.write(f"{qid}\t{docid}\t{score}\n")
    
    rank_output = f"{eval_output_dir}/{prefix}.{mode}.rank.tsv"
    generate_rank(output_file_path, rank_output)

    if mode == "dev":
        mrr = eval_results(rank_output)
        return mrr



def run_parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", choices=["train", "dev", "eval"], required=True)
    parser.add_argument("--output_dir", type=str, default=f"./data/train")
    
    parser.add_argument("--msmarco_dir", type=str, default=f"./data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--max_query_length", type=int, default=20)
    parser.add_argument("--max_doc_length", type=int, default=256)

    ## Other parameters
    parser.add_argument("--eval_ckpt", type=int, default=None)
    parser.add_argument("--per_gpu_eval_batch_size", default=26, type=int,)
    parser.add_argument("--per_gpu_train_batch_size", default=26, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--training_eval_steps", type=int, default=5000)

    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--data_num_workers", default=0, type=int)

    parser.add_argument("--learning_rate", default=3e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=10000, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1, type=int)

    args = parser.parse_args()

    time_stamp = time.strftime("%b-%d_%H:%M:%S", time.localtime())
    args.log_dir = f"{args.output_dir}/log/{time_stamp}"
    args.model_save_dir = f"{args.output_dir}/models"
    args.eval_save_dir = f"{args.output_dir}/eval_results"
    return args


def main():
    args = run_parse_args()
    logger.info(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    if args.task == "train":
        load_model_path = f"bert-base-uncased"
    else:
        assert args.eval_ckpt is not None
        load_model_path = f"{args.model_save_dir}/ckpt-{args.eval_ckpt}"
   

    config = BertConfig.from_pretrained(load_model_path)
    model = RepBERT_Train.from_pretrained(load_model_path, config=config)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    # Evaluation
    if args.task == "train":
        train(args, model)
    else:
        result = evaluate(args, model, args.task, prefix=f"ckpt-{args.eval_ckpt}")
        print(result)
    


if __name__ == "__main__":
    main()
