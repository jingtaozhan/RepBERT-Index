import math
import torch
import logging
from torch import nn
import numpy as np
from transformers.modeling_bert import (BertModel, BertPreTrainedModel)
logger = logging.getLogger(__name__)


def _average_query_doc_embeddings(sequence_output, token_type_ids, valid_mask):
    query_flags = (token_type_ids==0)*(valid_mask==1)
    doc_flags = (token_type_ids==1)*(valid_mask==1)

    query_lengths = torch.sum(query_flags, dim=-1)
    query_lengths = torch.clamp(query_lengths, 1, None)
    doc_lengths = torch.sum(doc_flags, dim=-1)
    doc_lengths = torch.clamp(doc_lengths, 1, None)
    
    query_embeddings = torch.sum(sequence_output * query_flags[:,:,None], dim=1)
    query_embeddings = query_embeddings/query_lengths[:, None]
    doc_embeddings = torch.sum(sequence_output * doc_flags[:,:,None], dim=1)
    doc_embeddings = doc_embeddings/doc_lengths[:, None]
    return query_embeddings, doc_embeddings


def _mask_both_directions(valid_mask, token_type_ids):
    assert valid_mask.dim() == 2
    attention_mask = valid_mask[:, None, :]

    type_attention_mask = torch.abs(token_type_ids[:, :, None] - token_type_ids[:, None, :])
    attention_mask = attention_mask - type_attention_mask
    attention_mask = torch.clamp(attention_mask, 0, None)
    return attention_mask


class RepBERT_Train(BertPreTrainedModel):
    def __init__(self, config):
        super(RepBERT_Train, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, valid_mask,
                position_ids, labels=None):
        attention_mask = _mask_both_directions(valid_mask, token_type_ids)

        sequence_output = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)[0]
        
        query_embeddings, doc_embeddings = _average_query_doc_embeddings(
            sequence_output, token_type_ids, valid_mask
        )
        
        similarities = torch.matmul(query_embeddings, doc_embeddings.T)
        
        output = (similarities, query_embeddings, doc_embeddings)
        if labels is not None:
            loss_fct = nn.MultiLabelMarginLoss()
            loss = loss_fct(similarities, labels)
            output = loss, *output
        return output
        


def _average_sequence_embeddings(sequence_output, valid_mask):
    flags = valid_mask==1
    lengths = torch.sum(flags, dim=-1)
    lengths = torch.clamp(lengths, 1, None)
    sequence_embeddings = torch.sum(sequence_output * flags[:,:,None], dim=1)
    sequence_embeddings = sequence_embeddings/lengths[:, None]
    return sequence_embeddings


class RepBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(RepBERT, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

        if config.encode_type == "doc":
            self.token_type_func = torch.ones_like
        elif config.encode_type == "query":
            self.token_type_func = torch.zeros_like
        else:
            raise NotImplementedError()
    def forward(self, input_ids, valid_mask):
        token_type_ids = self.token_type_func(input_ids)
        sequence_output = self.bert(input_ids,
                            attention_mask=valid_mask, 
                            token_type_ids=token_type_ids)[0]
        
        text_embeddings = _average_sequence_embeddings(
            sequence_output, valid_mask
        )
        
        return text_embeddings

    