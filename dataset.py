import os
import math
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from collections import namedtuple, defaultdict
from transformers import BertTokenizer
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)


class CollectionDataset:
    def __init__(self, collection_memmap_dir):
        self.pids = np.memmap(f"{collection_memmap_dir}/pids.memmap", dtype='int32',)
        self.lengths = np.memmap(f"{collection_memmap_dir}/lengths.memmap", dtype='int32',)
        self.collection_size = len(self.pids)
        self.token_ids = np.memmap(f"{collection_memmap_dir}/token_ids.memmap", 
                dtype='int32', shape=(self.collection_size, 512))
    
    def __len__(self):
        return self.collection_size

    def __getitem__(self, item):
        assert self.pids[item] == item
        return self.token_ids[item, :self.lengths[item]].tolist()


def load_queries(tokenize_dir, mode):
    queries = dict()
    for line in tqdm(open(f"{tokenize_dir}/queries.{mode}.json"), desc="queries"):
        data = json.loads(line)
        queries[int(data['id'])] = data['ids']
    return queries


def load_querydoc_pairs(msmarco_dir, mode):
    qrels = defaultdict(set)
    qids, pids, labels = [], [], []
    if mode == "train":
        for line in tqdm(open(f"{msmarco_dir}/qidpidtriples.train.small.tsv"),
                desc="load train triples"):
            qid, pos_pid, neg_pid = line.split("\t")
            qid, pos_pid, neg_pid = int(qid), int(pos_pid), int(neg_pid)
            qids.append(qid)
            pids.append(pos_pid)
            labels.append(1)
            qids.append(qid)
            pids.append(neg_pid)
            labels.append(0)
        for line in open(f"{msmarco_dir}/qrels.train.tsv"):
            qid, _, pid, _ = line.split()
            qrels[int(qid)].add(int(pid))
    else: 
        for line in open(f"{msmarco_dir}/top1000.{mode}"):
            qid, pid, _, _ = line.split("\t")
            qids.append(int(qid))
            pids.append(int(pid))
    qrels = dict(qrels)
    if not mode == "train":
        labels, qrels = None, None
    return qids, pids, labels, qrels


class MSMARCODataset(Dataset):
    def __init__(self, mode, msmarco_dir, 
            collection_memmap_dir, tokenize_dir,
            max_query_length=20, max_doc_length=256):

        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.qids, self.pids, self.labels, self.qrels = load_querydoc_pairs(msmarco_dir, mode)
        self.mode = mode
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

    def __len__(self):  
        return len(self.qids)

    def __getitem__(self, item):
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = [self.cls_id] + doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid" : pid
        }
        if self.mode == "train":
            ret_val["rel_docs"] = self.qrels[qid]
        return ret_val


def pack_tensor_2D(lstlst, default, dtype):
    batch_size = len(lstlst)
    length =  max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def get_collate_function(mode):
    def collate_function(batch):
        input_ids_lst = [x["query_input_ids"] + x["doc_input_ids"] for x in batch]
        token_type_ids_lst = [[0]*len(x["query_input_ids"]) + [1]*len(x["doc_input_ids"]) 
            for x in batch]
        valid_mask_lst = [[1]*len(input_ids) for input_ids in input_ids_lst]
        position_ids_lst = [list(range(len(x["query_input_ids"]))) + 
            list(range(len(x["doc_input_ids"]))) for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "valid_mask": pack_tensor_2D(valid_mask_lst, default=0, dtype=torch.int64),
            "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
        }
        qid_lst = [x['qid'] for x in batch]
        docid_lst = [x['docid'] for x in batch]
        if mode == "train":
            labels = [[j for j in range(len(docid_lst)) if docid_lst[j] in x['rel_docs'] ]for x in batch]
            data['labels'] =  pack_tensor_2D(labels, default=-1, dtype=torch.int64, length=len(batch))
        return data, qid_lst, docid_lst
    return collate_function  


def _test_dataset():
    dataset = MSMARCODataset(mode="train")
    for data in dataset:
        tokens = dataset.tokenizer.convert_ids_to_tokens(data["query_input_ids"])
        print(tokens)
        tokens = dataset.tokenizer.convert_ids_to_tokens(data["doc_input_ids"])
        print(tokens)
        print(data['qid'], data['docid'], data['rel_docs'])
        print()
        k = input()
        if k == "q":
            break


def _test_collate_func():
    from torch.utils.data import DataLoader, SequentialSampler
    eval_dataset = MSMARCODataset(mode="train")   
    train_sampler = SequentialSampler(eval_dataset)  
    collate_fn = get_collate_function(mode="train")
    dataloader = DataLoader(eval_dataset, batch_size=26,
        num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
    tokenizer = eval_dataset.tokenizer
    for batch, qidlst, pidlst in tqdm(dataloader):
        pass
        '''
        print(batch['input_ids'])
        print(batch['token_type_ids'])
        print(batch['valid_mask'])
        print(batch['position_ids'])
        print(batch['labels'])
        k = input()
        if k == "q":
            break
        '''

if __name__ == "__main__":
    _test_collate_func()
    

    
    