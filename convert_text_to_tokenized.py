import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer

def tokenize_file(tokenizer, input_file, output_file):
    total_size = sum(1 for _ in open(input_file))
    with open(output_file, 'w') as outFile:
        for line in tqdm(open(input_file), total=total_size,
                desc=f"Tokenize: {os.path.basename(input_file)}"):
            seq_id, text = line.split("\t")
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            outFile.write(json.dumps(
                {"id":seq_id, "ids":ids}
            ))
            outFile.write("\n")
    

def tokenize_queries(args, tokenizer):
    for mode in ["dev"]:#, "eval.small", "dev", "eval", "train"]:
        query_output = f"{args.output_dir}/queries.{mode}.json"
        tokenize_file(tokenizer, f"{args.msmarco_dir}/queries.{mode}.tsv", query_output)


def tokenize_collection(args, tokenizer):
    collection_output = f"{args.output_dir}/collection.tokenize.json"
    tokenize_file(tokenizer, f"{args.msmarco_dir}/collection.tsv", collection_output)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_dir", type=str, default="./data/msmarco-passage")
    parser.add_argument("--output_dir", type=str, default="./data/tokenize")
    parser.add_argument("--tokenize_queries", action="store_true")
    parser.add_argument("--tokenize_collection", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    if args.tokenize_queries:
        tokenize_queries(args, tokenizer)  
    if args.tokenize_collection:
        tokenize_collection(args, tokenizer) 
