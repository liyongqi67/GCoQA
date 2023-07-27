import os
import logging
import collections
import torch


from torch import nn





from transformers import T5Tokenizer, T5ForConditionalGeneration
logger = logging.getLogger(__name__)



class Generative_Retrieval(nn.Module):
    r"""
    
    """
    def __init__(self, args):
        super(Generative_Retrieval, self).__init__()
        self.generator = T5ForConditionalGeneration.from_pretrained(args.model_type, cache_dir=args.cache_dir)

    def forward(self, args=None, query_input_ids=None, query_attention_mask=None, 
                target_input_ids=None, target_attention_mask=None, 
                prefix_allowed_tokens_fn=None, mode="train"):
        if mode=="train":
            loss = self.generator(input_ids=query_input_ids, attention_mask=query_attention_mask, labels=target_input_ids).loss
            return loss
        if mode=="dev":
            outputs = self.generator.generate(query_input_ids,
                                  attention_mask= query_attention_mask,
                                  num_beams=5, 
                                  prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                  min_length=0,
                                  max_length=64,
                                  num_return_sequences =1)
            return outputs
        if mode=="test":
            outputs = self.generator.generate(query_input_ids,
                                  attention_mask= query_attention_mask,
                                  num_beams=args.beam_size, 
                                  prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                  min_length=0,
                                  max_length=64,
                                  num_return_sequences = args.top_k)
            return outputs
def dist_gather_tensor(t):
    if t is None:
        return None
    t = t.contiguous()

    all_tensors = [torch.empty_like(t) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_tensors, t)

    all_tensors[torch.distributed.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors

