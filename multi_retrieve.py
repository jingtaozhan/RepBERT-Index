import os
import math
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
import traceback
from functools import wraps
from queue import PriorityQueue
from multiprocessing import Pool, Manager
from retrieve import get_embed_memmap
from timeit import default_timer as timer


def raise_immediately(func):
    @wraps(func)
    def ret_func(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            print(traceback.format_exc())
            raise
    return ret_func


@raise_immediately
def writer(args, finish_queue_lst):
    _, query_id_memmap = get_embed_memmap(
        args.query_embedding_dir, args.embedding_dim)
    with open(args.output_path, 'w') as outFile:
        for qid in query_id_memmap:
            score_docid_lst = []
            for q in finish_queue_lst:
                score_docid_lst = score_docid_lst + q.get()
            score_docid_lst = sorted(score_docid_lst, reverse=True)
            for rank_idx, (score, para_id) in enumerate(score_docid_lst[:args.hit]):
                outFile.write(f"{qid}\t{para_id}\t{rank_idx+1}\n")


@raise_immediately
def allrank(gpu_queue, doc_begin_index, doc_end_index, finish_queue):
    import os
    import torch
    gpuid = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{gpuid}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.device_count() == 1

    query_embedding_memmap, query_id_memmap = get_embed_memmap(
        args.query_embedding_dir, args.embedding_dim)
    qid2pos = {identity:i for i, identity in enumerate(query_id_memmap)}

    doc_embedding_memmap, doc_id_memmap = get_embed_memmap(
        args.doc_embedding_dir, args.embedding_dim)
    assert np.all(doc_id_memmap == list(range(len(doc_id_memmap))))

    doc_embeddings = doc_embedding_memmap[doc_begin_index:doc_end_index]
    doc_ids = doc_id_memmap[doc_begin_index:doc_end_index]

    doc_embeddings = torch.from_numpy(doc_embeddings).to(device)
    results_dict = {qid:PriorityQueue(maxsize=args.hit) for qid in query_id_memmap}

    for qid in tqdm(query_id_memmap, desc=f"{gpuid}"):
        query_embedding = query_embedding_memmap[qid2pos[qid]]
        query_embedding = torch.from_numpy(query_embedding)
        query_embedding = query_embedding.to(device)
    
        all_scores = torch.sum(query_embedding * doc_embeddings, dim=-1)
        
        k = min(args.hit, len(doc_embeddings))
        top_scores, top_indices = torch.topk(all_scores, k, largest=True, sorted=True)
        top_scores, top_indices = top_scores.cpu(), top_indices.cpu()
        top_doc_ids = doc_ids[top_indices.numpy()]
        cur_q_queue = results_dict[qid]
        for score, docid in zip(top_scores, top_doc_ids):
            score, docid = score.item(), docid.item()
            if cur_q_queue.full():
                lowest_score, lowest_docid = cur_q_queue.get_nowait()
                if lowest_score >= score:
                    cur_q_queue.put_nowait((lowest_score, lowest_docid))
                    break
                else:
                    cur_q_queue.put_nowait((score, docid))
            else:
                cur_q_queue.put_nowait((score, docid))
        finish_queue.put(cur_q_queue.queue)
    doc_embeddings, all_scores, query_embedding, top_scores, top_indices = None, None, None, None, None
    torch.cuda.empty_cache()
    gpu_queue.put_nowait(gpuid)


if __name__ == "__main__":
    work_dir = "/home/zhanjingtao/workspace/repbert"
    output_root = f"{work_dir}/msmarco/data/first_stage"
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--per_gpu_doc_num", type=int, default=None)
    parser.add_argument("--hit", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--output_path", type=str, 
        default="./data/retrieve/repbert.dev.small.top1k.tsv")
    parser.add_argument("--doc_embedding_dir", type=str, 
        default="./data/precompute/doc_embedding")
    parser.add_argument("--query_embedding_dir", type=str, 
        default="./data/precompute/query_dev.small_embedding")
    
    args = parser.parse_args()

    doc_size = len(get_embed_memmap(args.doc_embedding_dir, args.embedding_dim)[1])
    if args.per_gpu_doc_num is None:
        args.per_gpu_doc_num = math.ceil(doc_size / len(args.gpus))

    num_rounds = math.ceil(doc_size / args.per_gpu_doc_num)
    doc_arguments = []
    for i in range(num_rounds):
        doc_begin_index = int(doc_size *  i / num_rounds)
        doc_end_index = int(doc_size * (i+1) / num_rounds)
        doc_arguments.append((doc_begin_index, doc_end_index))

    manager = Manager()
    finished_queue_lst = [manager.Queue() for _ in range(num_rounds)]
    gpu_queue = manager.Queue()
    for gpu in args.gpus:
        gpu_queue.put_nowait(gpu)

    pool = Pool(num_rounds+1)
    start = timer()
    for finish_queue, (doc_begin_index, doc_end_index) in zip(finished_queue_lst, doc_arguments):
        pool.apply_async(allrank, 
            args=(gpu_queue, doc_begin_index, doc_end_index, finish_queue))
    pool.apply_async(writer, args=(args, finished_queue_lst))
    pool.close()
    pool.join()
    end = timer()
    print(end - start)