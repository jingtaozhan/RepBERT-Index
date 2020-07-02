# RepBERT

RepBERT is is currently the state-of-the-art first-stage retrieval technique on [MS MARCO Passage Ranking task](https://microsoft.github.io/msmarco/). It represents documents and queries with fixed-length contextualized embeddings. The inner products of them are regarded as relevance scores. Its efficiency is comparable to bag-of-words methods. For more details, check out our paper:

+ Zhan et al.  [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval.](https://arxiv.org/abs/2006.15498)


MS MARCO Passage Ranking Leaderboard (Jun 28th 2020) | Eval MRR@10 | Latency
:------------------------------------ | :------: | ------:
[BM25 + BERT](https://github.com/nyu-dl/dl4marco-bert) from [(Nogueira and Cho, 2019)](https://arxiv.org/abs/1901.04085) | 0.358 | 3400 ms
BiLSTM + Co-Attention + self attention based document scorer [(Alaparthi et al., 2019)](https://arxiv.org/abs/1906.06056) (best non-BERT) | 0.291 | -
RepBERT (this code)             | 0.294 | 66 ms
[docTTTTTquery](https://github.com/castorini/docTTTTTquery) [(Nogueira1 et al., 2019)](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery.pdf)             | 0.272 | 64 ms
[DeepCT](https://github.com/AdeDZY/DeepCT) [(Dai and Callan, 2019)](https://github.com/AdeDZY/DeepCT)              | 0.239 | 55 ms
[doc2query](https://github.com/nyu-dl/dl4ir-doc2query) [(Nogueira et al., 2019)](https://github.com/nyu-dl/dl4ir-doc2query)              | 0.218 | 90 ms
[BM25(Anserini)](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)  | 0.186  | 50 ms


## Data and Trained Models

We make the following data available for download:

+ `repbert.dev.small.top1k.tsv`: 6,980,000 pairs of dev set queries and retrieved passages. In this tsv file, the first column is the query id, the second column is the passage id, and the third column is the rank of the passage. There are 1000 passages per query in this file.
+ `repbert.eval.small.top1k.tsv`: 6,837,000 pairs of eval set queries and retrieved passages. In this tsv file, the first column is the query id, the second column is the passage id, and the third column is the rank of the passage. There are 1000 passages per query in this file.
+ `repbert.ckpt-350000.zip`: Trained BERT base model to represent queries and passages. It contains two files, namely `config.json` and `pytorch_model.bin`.

Download and verify the above files from the below table:

File | Size | MD5 | Download
:----|-----:|:----|:-----
`repbert.dev.small.top1k.tsv` | 127 MB | `0d08617b62a777c3c8b2d42ca5e89a8e` | [[Google Drive](https://drive.google.com/file/d/1MrrwDmTZOiFx3qjfPxi4lDSdQk1tR5C6/view?usp=sharing)]
`repbert.eval.small.top1k.tsv` | 125 MB | `b56a79138f215292d674f58c694d5206` | [[Google Drive](https://drive.google.com/file/d/1twRGEJZFZc4zYa75q8UFEz9ZS2oh0oyE/view?usp=sharing)]
`repbert.ckpt-350000.zip` | 386 MB| `b59a574f53c92de6a4ddd4b3fbef784a` | [[Google Drive](https://drive.google.com/file/d/1xhwy_nvRWSNyJ2V7uP3FC5zVwj1Xmylv/view?usp=sharing)] 


## Replicating Results with Provided Trained Model

We provide instructions on how to replicate RepBERT retrieval results using provided trained model.

First, make sure you already installed [🤗 Transformers](https://github.com/huggingface/transformers):

```bash
pip install transformers
git clone https://github.com/jingtaozhan/RepBERT-Index
cd RepBERT-Index
```

Next, download `collectionandqueries.tar.gz` from [MSMARCO-Passage-Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking). It contains passages, queries, and qrels.

```bash
mkdir data
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
mkdir msmarco-passage
tar xvfz collectionandqueries.tar.gz -C msmarco-passage
```

To confirm, `collectionandqueries.tar.gz` should have MD5 checksum of `31644046b18952c1386cd4564ba2ae69`.

To reduce duplication of effort in training and testing, we tokenize queries and passages in advance. This should take some time (about 3-4 hours). Besides, we convert tokenized passages to numpy memmap array, which can greatly reduce the memory overhead at run time.

```bash
python convert_text_to_tokenized.py --tokenize_queries --tokenize_collection
python convert_collection_to_memmap.py
```

Please download the provided model `repbert.ckpt-350000.zip`, put it in `./data`, and unzip it. You should see two files in the directory `./data/ckpt-350000`, namely `pytorch_model.bin` and `config.json`.

Next, you need to precompute the representations of passages and queries. Strictly speaking, the representations of queries should not be computed in advance (offline). In our implementation, we do it to keep the code elegant. Note that it takes little time (less than 1 ms/query) to compute the representations of queries. Therefore, precomputing them hardly affects the results of query latency in the retrieval stage. 

```bash
python precompute.py --load_model_path ./data/ckpt-350000 --task doc
python precompute.py --load_model_path ./data/ckpt-350000 --task query_dev.small
python precompute.py --load_model_path ./data/ckpt-350000 --task query_eval.small
```

At last, you can retrieve the passages for the queries in the dev set (or eval set). `multi_retrieve.py` will use the gpus specified by `--gpus` argument and the representations of all passages are evenly distributed among all gpus. If your CUDA memory is limited, you can use `--per_gpu_doc_num` to specify the num of passages distributed to each gpu. For example, `--per_gpu_doc_num` can be set to `1800000` for a conventional 12-GB gpu.

```bash
python multi_retrieve.py  --query_embedding_dir ./data/precompute/query_dev.small_embedding --output_path ./data/retrieve/repbert.dev.small.top1k.tsv --hit 1000 --gpus 0,1,2,3,4
python ./ms_marco_eval.py ./data/msmarco-passage/qrels.dev.small.tsv ./data/retrieve/repbert.dev.small.top1k.tsv
```

The results should be:

```
#####################
MRR @10: 0.3038783713103188
QueriesRanked: 6980
#####################
```

## Train RepBERT

The code and instructions will be released soon.
