from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
import linecache
import numpy as np
from io import open
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
import csv
from datasets import load_dataset


import random
from random import choice

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
# from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)



class FinetuningDataset(Dataset):
    def __init__(self, filename, tokenizer,
                 load_small, query_max_seq_length,target_max_seq_length, prepend_answers):
        
        self._filename = filename
        self._tokenizer = tokenizer
        self._load_small = load_small

        self._query_max_seq_length = query_max_seq_length

        self._target_max_seq_length = target_max_seq_length

        self.data = []
        with open(filename, 'r') as load_f:
            data = json.load(load_f)
            for entry in data:
                if entry['Answer'] != "UNANSWERABLE":
                    self.data.append(entry)
                # if entry['Answer'] == "UNANSWERABLE":       ###############for answer generation experiemnt
                #     entry['Page'] = 'The Young and the Restless'
                #     entry['Passage'] = {
                #                         "id": 9185426,
                #                             "title": "The Young and the Restless [SEP] Casting and story development",
                #                             "text": "Co-creators William J. Bell and Lee Phillip Bell centered The Young and the Restless around two core families, the wealthy Brooks and the poor Fosters. Bell borrowed this technique of soap opera building from his mentor, Irna Phillips. While casting for the series, Bell and executive producer John Conboy auditioned 540 actors for the 13 main characters. They assembled the youngest group of actors ever cast on a soap opera at the time, hiring mostly unknown actors whom they considered glamorous model types. Chemistry between actors also factored into the criteria for casting. The stories focused on the younger characters, with an emphasis in fantasy. The fantasy element was reflected in the love story between Jill Abbott and the millionaire Phillip Chancellor II; the Leslie Brooks, Brad Elliot, and Lorie Brooks love triangle; and Snapper Fosters romance with Chris Brooks. Sexuality also played a major role in the stories. Formerly, soap operas did not delve into the sexual side of their romances. Bell changed that, first during his time as head writer of Days of Our Lives and again on The Young and the Restless. William Gray Espys Snapper Foster is considered the first to discover sex on a soap opera. During the story, the character is engaged to Chris Brooks (Trish Stewart) and having a sexual relationship with Sally McGuire (Lee Crawford). Other plots reflected sexual themes as well. For the first time in the genre, the dialogue and the story situations included explicit sexual themes such as premarital intercourse, impotence, incest, and rape. The first two rape storylines that would be told on the serial were controversial at the time as they reflected a more introspective and analytic storytelling style, the first time rape storylines would be addressed in this manner in the genre. The first, in 1973–74, revolved around the rape of Chris Brooks and the aftermath, in which she entertained (and, eventually, rejected) the idea that she was perhaps at fault for her attack. The second, in 1976, involved Chriss sister Peggy (Pamela Peters Solow) and was meant to serve as a cut-and-dried story in which no viewer could justify this attack, committed out of the blue by an authority figure."
                #                         }
                #     self.data.append(entry)
        # if 'train' in filename:
        #     random.shuffle(self.data)
        #     self.data = self.data[:int(len(self.data)*0.0)]

        self._total_data = 0      
        if self._load_small:
            self._total_data = 100
        else:
            self._total_data = len(self.data)

        self.prepend_answers = prepend_answers
    def __len__(self):
        return self._total_data
                
    def __getitem__(self, idx):
  
        entry = self.data[idx]

        if self.prepend_answers:
            entry['Question'] = " [SEP] ".join(entry['Context'])+ " [SEP] " +  entry['Question']
        else:
            s = []
            for i in range(len(entry['Context'])):
                if i%2 == 0:
                    s.append(entry['Context'][i])
            entry['Question'] = " [SEP] ".join(s)+ " [SEP] " +  entry['Question']

        query_feature = text_to_feature(entry['Question'], self._tokenizer, 
                                max_length=self._query_max_seq_length)  

        target_text = entry["Passage"]['title'].strip()


        target_feature = text_to_feature(target_text, self._tokenizer, 
                                max_length=self._target_max_seq_length)
        return_feature_dict = {   'query_input_ids': np.asarray(query_feature['input_ids']), 
                                  'query_attention_mask': np.asarray(query_feature['attention_mask']),
                                  'query_text': entry['Question'],
                                  'target_input_ids': np.asarray(target_feature['input_ids']), 
                                  'target_attention_mask': np.asarray(target_feature['attention_mask']),
                                  'target_text': target_text,
                                  'answer_text': entry["Answer"],
                                }           

        return return_feature_dict


def text_to_feature(text, tokenizer,
                                      max_length=256,                                      
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):

    input_ids  = tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True
    )

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)

    inputs = {}
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask


    return inputs





def normalize_question(question: str) -> str:
    return question

def normalize_passage(ctx_text: str):
    return ctx_text