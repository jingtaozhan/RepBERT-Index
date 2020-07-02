import os
import math
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from collections import namedtuple, defaultdict
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
from dataset import (load_querydoc_pairs, load_queries, CollectionDataset, pack_tensor_2D, MSMARCODataset)
from modeling import RepBERT

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)


def create_embed_memmap(ids, memmap_dir, dim):
    if not os.path.exists(memmap_dir):
        os.makedirs(memmap_dir)
    embedding_path = f"{memmap_dir}/embedding.memmap"
    id_path = f"{memmap_dir}/ids.memmap"
    embed_open_mode = "r+" if os.path.exists(embedding_path) else "w+"
    id_open_mode = "r+" if  os.path.exists(id_path) else "w+"
    logger.warning(f"Open Mode: embedding-{embed_open_mode} ids-{id_open_mode}")

    embedding_memmap = np.memmap(embedding_path, dtype='float32', 
        mode=embed_open_mode, shape=(len(ids), dim))
    id_memmap = np.memmap(id_path, dtype='int32', 
        mode=id_open_mode, shape=(len(ids),))
    id_memmap[:] = ids
    # not writable
    id_memmap = np.memmap(id_path, dtype='int32', 
        shape=(len(ids),))
    return embedding_memmap, id_memmap


class MSMARCO_QueryDataset(Dataset):
    def __init__(self, tokenize_dir, msmarco_dir, task, max_query_length):
        self.max_query_length = max_query_length
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.queries = load_queries(tokenize_dir, task)
        self.qids = list(self.queries.keys())
        self.task = task
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.all_ids = self.qids

    def __len__(self):  
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        query_input_ids = self.queries[qid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        ret_val = {
            "input_ids": query_input_ids,
            "id" : qid
        }
        return ret_val


class MSMARCO_DocDataset(Dataset):
    def __init__(self, collection_memmap_dir, max_doc_length):
        self.max_doc_length = max_doc_length
        self.collection = CollectionDataset(collection_memmap_dir)
        self.pids = self.collection.pids
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.all_ids = self.collection.pids

    def __len__(self):  
        return len(self.pids)

    def __getitem__(self, item):
        pid = self.pids[item]
        doc_input_ids = self.collection[pid]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = [self.cls_id] + doc_input_ids + [self.sep_id]

        ret_val = {
            "input_ids": doc_input_ids,
            "id" : pid
        }
        return ret_val


def get_collate_function():
    def collate_function(batch):
        input_ids_lst = [x["input_ids"] for x in batch]
        valid_mask_lst = [[1]*len(input_ids) for input_ids in input_ids_lst]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, 
                dtype=torch.int64),
            "valid_mask": pack_tensor_2D(valid_mask_lst, default=0, 
                dtype=torch.int64),
        }
        id_lst = [x['id'] for x in batch]
        return data, id_lst
    return collate_function  


def generate_embeddings(args, model, task):
    if task == "doc":
        dataset = MSMARCO_DocDataset(args.collection_memmap_dir, args.max_doc_length)
        memmap_dir = args.doc_embedding_dir
    else: 
        query_str, mode = task.split("_")
        assert query_str == "query"
        dataset = MSMARCO_QueryDataset(args.tokenize_dir, args.msmarco_dir, mode, args.max_query_length)
        memmap_dir = args.query_embedding_dir
    embedding_memmap, ids_memmap = create_embed_memmap(
        dataset.all_ids, memmap_dir, model.config.hidden_size)
    id2pos = {identity:i for i, identity in enumerate(ids_memmap)}
    
    batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    collate_fn = get_collate_function()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)

    start = timer()
    for batch, ids in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            batch = {k:v.to(args.device) for k, v in batch.items()}
            output = model(**batch)
            sequence_embeddings = output.detach().cpu().numpy()
            poses = [id2pos[identity] for identity in ids]
            embedding_memmap[poses] = sequence_embeddings
    end = timer()
    print(task, "time:", end-start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--load_model_path", type=str, required=True)
    parser.add_argument("--task", choices=["query_dev.small", "query_eval.small", "doc"],
        required=True)
    parser.add_argument("--output_dir", type=str, default="./data/precompute")

    parser.add_argument("--msmarco_dir", type=str, default=f"./data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--max_query_length", type=int, default=20)
    parser.add_argument("--max_doc_length", type=int, default=256)
    parser.add_argument("--per_gpu_batch_size", default=100, type=int)
    args = parser.parse_args()

    args.doc_embedding_dir = f"{args.output_dir}/doc_embedding"
    args.query_embedding_dir = f"{args.output_dir}/{args.task}_embedding"

    logger.info(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    config = BertConfig.from_pretrained(args.load_model_path)
    if "query" in args.task:
        config.encode_type = "query"
    else:
        config.encode_type = "doc"
    model = RepBERT.from_pretrained(args.load_model_path, config=config)
    model.to(args.device)

    logger.info(args)
    generate_embeddings(args, model, args.task)


    
