# RepBERT-Index

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
+ `repbert.ckpt-350000.pytorch.bin`: Trained BERT base model to represent queries and passages.

Download and verify the above files from the below table:

File | Size | MD5 | Download
:----|-----:|:----|:-----
`repbert.dev.small.top1k.tsv` | 127 MB | `0d08617b62a777c3c8b2d42ca5e89a8e` | [[Google Drive](https://drive.google.com/file/d/1MrrwDmTZOiFx3qjfPxi4lDSdQk1tR5C6/view?usp=sharing)]
`repbert.eval.small.top1k.tsv` | 125 MB | `b56a79138f215292d674f58c694d5206` | [[Google Drive](https://drive.google.com/file/d/1twRGEJZFZc4zYa75q8UFEz9ZS2oh0oyE/view?usp=sharing)]
`repbert.ckpt-350000.pytorch.bin` | 418 MB| `bc1f80e34a88e5277b8977a931953e26` | [[Google Drive](https://drive.google.com/file/d/1pw7_bc8B664hzYaseBqorYzxirIL2-qt/view?usp=sharing)] 
