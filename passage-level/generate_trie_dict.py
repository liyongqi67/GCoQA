from datasets import load_dataset
from tqdm import tqdm
import re
import json
from transformers import T5Tokenizer
from utils import Trie
import pickle

tokenizer = T5Tokenizer.from_pretrained("t5-base", do_lower_case=True,
                                                          cache_dir="/home/v-yongqili/project/GCoQA/data/huggingface_cache/")
page_title_dict = {}
with open("/home/v-yongqili/project/GCoQA/data/full_wiki_segments.json", 'r') as f:
    data = f.readlines()
    for line in tqdm(data):
        line = json.loads(line)
        if line['title'].strip() not in page_title_dict:
            page_title_dict[line['title'].strip()] = 1

print("page_title_dict len %s", len(page_title_dict))

title_sequence = []
for page_title in tqdm(page_title_dict):
    input_ids = tokenizer.encode(
    page_title,
    add_special_tokens=True,
    max_length=64,
    truncation=True)
    title_sequence.append([0] + input_ids)

decoder_trie = Trie(title_sequence)
with open("/home/v-yongqili/project/GCoQA/data/trie_dict_t5-base_section_level.pkl", 'wb') as f:
    pickle.dump(decoder_trie.trie_dict, f)