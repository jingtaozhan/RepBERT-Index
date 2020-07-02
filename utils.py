import os
import re
import random
from collections import defaultdict
import subprocess


def generate_rank(input_path, output_path):
    score_dict = defaultdict(list)
    for line in open(input_path):
        query_id, para_id, score = line.split("\t")
        score_dict[int(query_id)].append((float(score), int(para_id)))
    with open(output_path, "w") as outFile:
        for query_id, para_lst in score_dict.items():
            random.shuffle(para_lst)
            para_lst = sorted(para_lst, key=lambda x:x[0], reverse=True)
            for rank_idx, (score, para_id) in enumerate(para_lst):
                outFile.write("{}\t{}\t{}\n".format(query_id, para_id, rank_idx+1))


def eval_results(run_file_path,
        eval_script="./ms_marco_eval.py", 
        qrels="./data/msmarco-passage/qrels.dev.small.tsv" ):
    assert os.path.exists(eval_script) and os.path.exists(qrels)
    result = subprocess.check_output(['python', eval_script, qrels, run_file_path])
    match = re.search('MRR @10: ([\d.]+)', result.decode('utf-8'))
    mrr = float(match.group(1))
    return mrr
