import os
import math
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from queue import PriorityQueue
from collections import namedtuple, defaultdict
from transformers import BertTokenizer, BertConfig
from torch.utils.data import DataLoader, Dataset
from dataset import CollectionDataset, pack_tensor_2D, MSMARCODataset
from utils import generate_rank, eval_results

def get_embed_memmap(memmap_dir, dim):
    embedding_path = f"{memmap_dir}/embedding.memmap"
    id_path = f"{memmap_dir}/ids.memmap"
    # Tensor doesn't support non-writeable numpy array
    # Thus we use copy-on-write mode 
    id_memmap = np.memmap(id_path, dtype='int32', mode="c")
    embedding_memmap = np.memmap(embedding_path, dtype='float32', 
        mode="c", shape=(len(id_memmap), dim))
    return embedding_memmap, id_memmap


def allrank(args):
    doc_embedding_memmap, doc_id_memmap = get_embed_memmap(
        args.doc_embedding_dir, args.embedding_dim)
    assert np.all(doc_id_memmap == list(range(len(doc_id_memmap))))

    query_embedding_memmap, query_id_memmap = get_embed_memmap(
        args.query_embedding_dir, args.embedding_dim)
    qid2pos = {identity:i for i, identity in enumerate(query_id_memmap)}
    results_dict = {qid:PriorityQueue(maxsize=args.hit) for qid in query_id_memmap}

    for doc_begin_index in tqdm(range(0, len(doc_id_memmap), args.per_gpu_doc_num), desc="doc"):
        doc_end_index = doc_begin_index+args.per_gpu_doc_num
        doc_ids = doc_id_memmap[doc_begin_index:doc_end_index]
        doc_embeddings = doc_embedding_memmap[doc_begin_index:doc_end_index]
        doc_embeddings = torch.from_numpy(doc_embeddings).to(args.device)
        for qid in tqdm(query_id_memmap, desc="query"):
            query_embedding = query_embedding_memmap[qid2pos[qid]]
            query_embedding = torch.from_numpy(query_embedding)
            query_embedding = query_embedding.to(args.device)
        
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

    score_path = f"{args.output_path}.score"    
    with open(score_path, 'w') as outputfile:
        for qid, docqueue in results_dict.items():
            while not docqueue.empty():
                score, docid = docqueue.get_nowait()
                outputfile.write(f"{qid}\t{docid}\t{score}\n")
    generate_rank(score_path, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--per_gpu_doc_num", default=1800000, type=int)
    parser.add_argument("--hit", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--output_path", type=str, 
        default="./data/retrieve/repbert.dev.small.top1k.tsv")
    parser.add_argument("--doc_embedding_dir", type=str, 
        default="./data/precompute/doc_embedding")
    parser.add_argument("--query_embedding_dir", type=str, 
        default="./data/precompute/query_dev.small_embedding")
    args = parser.parse_args()

    print(args)

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    assert args.n_gpu == 1

    args.device = device

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    with torch.no_grad():
        allrank(args)