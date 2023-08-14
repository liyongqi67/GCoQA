# GCoQA
This is the official implementation for the paper "Generative Retrieval for Conversational Question Answering".  
The paper is released in [link](https://liyongqi67.github.io/papers/Generative%20Retrieval%20for%20Conversational%20Question%20Answering.pdf).  
If you find our paper or code helpful,please consider citing as follows:
```bibtex
@article{LI2023103475,
title = {Generative retrieval for conversational question answering},
journal = {Information Processing & Management},
volume = {60},
number = {5},
pages = {103475},
year = {2023},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2023.103475},
url = {https://www.sciencedirect.com/science/article/pii/S0306457323002121},
author = {Yongqi Li and Nan Yang and Liang Wang and Furu Wei and Wenjie Li},
}
```

## Dataset
We conducted experiments on three conversational open-domain QA datasets: OR-QuAC, QRECC, and TOPIOCQA. To facilitate future research in this area, we unified the three datasets into a benchmark with the same corpus, as DPR did.  
### 1. Corpus.  
1.1 Passage-level corpus: full_wiki_segments.json.  
Format:
```
{
'id': 0,
'title': 'Eliza Fletcher [SEP] Introduction',
'text': 'Eliza Fletcher, née Dawson (15 January 1770 – 5 February 1858) was an English autobiographer and early travel writer.'
}
```
"Eliza Fletcher" is the page title, and "Introduction" is the section title, in Wikipedia.  
1.2 Document-level corpus: full_wiki_document.json  
Format:
```
{
'id': 0,
'title': 'Eliza Fletcher',
'text': '......'
}
```
"Eliza Fletcher" is the page title in Wikipedia.  
### 2. QA pairs.  
TOPIOCQA dataset: topiocqa_train.json, topiocqa_dev.json, topiocqa_test.json.    
QRECC dataset: qrecc_train.json, qrecc_dev.json, qrecc_test.json.  
OR-QUAC dataset: orquac_train.json, orquac_dev.json, orquac_test.json.  
Format:
```
 {
    "Conversation_no": 3209,
    "Turn_no": 2,
    "Context": [
      "who is  finn m. w. caspersen?",
      "American financier and philanthropist."
    ],
    "Question": "where did he study?",
    "Gold_question": "",
    "Answer": "Peddie School, Brown University, and Harvard Law School.",
    "Page": "Finn M. W. Caspersen",
    "Section": "Early life and education",
    "Passage": {
      "id": "8114812",
      "text": "He later reflected that being Protestant was important. There was a kind of anti-Catholicism in the family. The family moved to homes in Andover, New Jersey, and Venice, Florida. Caspersen frequently visited Norway as a child, vacationing there during summers after 1947. Caspersen attended private schools until the ninth grade. He attended the Peddie School, a private preparatory school in Hightstown, New Jersey, and was graduated in 1959. Caspersen received a Bachelor of Arts (B.A.) degree from Brown University in 1963 and a law degree (LL.B.) from Harvard Law School in 1966.",
      "title": "Finn M. W. Caspersen [SEP] Early life and education"
    }
  }
```
### 3. Trie. 
To implement the constrained generation in the LLM, we process all the corpus and store it in the trie structure.  
You could use the scripts passage-level/generate_trie_dict.py and document-level/generate_trie_dict.py to obtain the trie for passages and documents, respectively.  
You could also download our processed trie files.  
```
trie_dict_t5-base_section_level.pkl is for the passage_level.  
trie_dict_t5-base_page_level.pkl is for the document_level.
```
### 4. Download. 
You could download the above files via this [link](https://drive.google.com/drive/folders/18Sa7QPO0r6j-OSVdoiobzAqcADIn4cfM?usp=sharing).

## Model training  
### Passage_level
The script for training on the TOPIOCQA dataset is 
```bash
    - python3 -m torch.distributed.launch --nproc_per_node 8 passage-level/train_query_encoder.py
      --do_train True
      --load_small False
      --fp16 False
      --num_train_epochs 40
      --per_gpu_train_batch_size 8
      --per_gpu_eval_batch_size 4
      --per_gpu_test_batch_size 2
      --overwrite_output_dir True
      --train_file $$AMLT_DATA_DIR/QA_pairs/topiocqa/topiocqa_train.json
      --dev_file $$AMLT_DATA_DIR/QA_pairs/topiocqa/topiocqa_dev.json
      --test_file $$AMLT_DATA_DIR/QA_pairs/topiocqa/topiocqa_test.json
      --corpus_path $$AMLT_DATA_DIR/full_wiki_segments.json
      --cache_dir $$AMLT_DATA_DIR/huggingface_cache/
      --trie_dict $$AMLT_DATA_DIR/trie_dict_t5-base_section_level.pkl
      --output_dir $$AMLT_OUTPUT_DIR/release_test/
      --learning_rate 1e-5
      --prepend_answers True
      --model_type t5-large
      --top_k 5
      --beam_size 5
```
The script for training on the QRECC dataset is 
```bash
    - python3 -m torch.distributed.launch --nproc_per_node 8 passage-level/train_query_encoder.py
      --do_train True
      --load_small False
      --fp16 False
      --num_train_epochs 40
      --per_gpu_train_batch_size 8
      --per_gpu_eval_batch_size 4
      --per_gpu_test_batch_size 2
      --overwrite_output_dir True
      --train_file $$AMLT_DATA_DIR/QA_pairs/qrecc/qrecc_train.json
      --dev_file $$AMLT_DATA_DIR/QA_pairs/qrecc/qrecc_dev.json
      --test_file $$AMLT_DATA_DIR/QA_pairs/qrecc/qrecc_test.json
      --corpus_path $$AMLT_DATA_DIR/full_wiki_segments.json
      --cache_dir $$AMLT_DATA_DIR/huggingface_cache/
      --trie_dict $$AMLT_DATA_DIR/trie_dict_t5-base_section_level.pkl
      --output_dir $$AMLT_OUTPUT_DIR/release_test/
      --learning_rate 1e-5
      --prepend_answers True
      --model_type t5-large
      --top_k 5
      --beam_size 5
```
The script for training on the OR-QUAC dataset is 
```bash
    - python3 -m torch.distributed.launch --nproc_per_node 8 passage-level/train_query_encoder.py
      --do_train True
      --load_small False
      --fp16 False
      --num_train_epochs 40
      --per_gpu_train_batch_size 8
      --per_gpu_eval_batch_size 4
      --per_gpu_test_batch_size 2
      --overwrite_output_dir True
      --train_file $$AMLT_DATA_DIR/QA_pairs/orquac/orquac_train.json
      --dev_file $$AMLT_DATA_DIR/QA_pairs/orquac/orquac_dev.json
      --test_file $$AMLT_DATA_DIR/QA_pairs/orquac/orquac_test.json
      --corpus_path $$AMLT_DATA_DIR/full_wiki_segments.json
      --cache_dir $$AMLT_DATA_DIR/huggingface_cache/
      --trie_dict $$AMLT_DATA_DIR/trie_dict_t5-base_section_level.pkl
      --output_dir $$AMLT_OUTPUT_DIR/release_test/
      --learning_rate 1e-5
      --prepend_answers False
      --model_type t5-large
      --top_k 5
      --beam_size 5
``` 
### Document_level
The script for training on the TOPIOCQA dataset is 
```bash
    - python3 -m torch.distributed.launch --nproc_per_node 8 document-level/train_query_encoder.py
      --do_train True
      --load_small False
      --fp16 False
      --num_train_epochs 40
      --per_gpu_train_batch_size 8
      --per_gpu_eval_batch_size 4
      --per_gpu_test_batch_size 2
      --overwrite_output_dir True
      --train_file $$AMLT_DATA_DIR/QA_pairs/topiocqa/topiocqa_train.json
      --dev_file $$AMLT_DATA_DIR/QA_pairs/topiocqa/topiocqa_dev.json
      --test_file $$AMLT_DATA_DIR/QA_pairs/topiocqa/topiocqa_test.json
      --corpus_path $$AMLT_DATA_DIR/full_wiki_document.json
      --cache_dir $$AMLT_DATA_DIR/huggingface_cache/
      --trie_dict $$AMLT_DATA_DIR/trie_dict_t5-base_page_level.pkl
      --output_dir $$AMLT_OUTPUT_DIR/release_test/
      --learning_rate 1e-5
      --prepend_answers True
      --model_type t5-large
      --top_k 5
      --beam_size 5
```
The script for training on the QRECC dataset is 
```bash
    - python3 -m torch.distributed.launch --nproc_per_node 8 document-level/train_query_encoder.py
      --do_train True
      --load_small False
      --fp16 False
      --num_train_epochs 40
      --per_gpu_train_batch_size 8
      --per_gpu_eval_batch_size 4
      --per_gpu_test_batch_size 2
      --overwrite_output_dir True
      --train_file $$AMLT_DATA_DIR/QA_pairs/qrecc/qrecc_train.json
      --dev_file $$AMLT_DATA_DIR/QA_pairs/qrecc/qrecc_dev.json
      --test_file $$AMLT_DATA_DIR/QA_pairs/qrecc/qrecc_test.json
      --corpus_path $$AMLT_DATA_DIR/full_wiki_document.json
      --cache_dir $$AMLT_DATA_DIR/huggingface_cache/
      --trie_dict $$AMLT_DATA_DIR/trie_dict_t5-base_page_level.pkl
      --output_dir $$AMLT_OUTPUT_DIR/release_test/
      --learning_rate 1e-5
      --prepend_answers True
      --model_type t5-large
      --top_k 5
      --beam_size 5
```
The script for training on the OR-QUAC dataset is 
```bash
    - python3 -m torch.distributed.launch --nproc_per_node 8 document-level/train_query_encoder.py
      --do_train True
      --load_small False
      --fp16 False
      --num_train_epochs 40
      --per_gpu_train_batch_size 8
      --per_gpu_eval_batch_size 4
      --per_gpu_test_batch_size 2
      --overwrite_output_dir True
      --train_file $$AMLT_DATA_DIR/QA_pairs/orquac/orquac_train.json
      --dev_file $$AMLT_DATA_DIR/QA_pairs/orquac/orquac_dev.json
      --test_file $$AMLT_DATA_DIR/QA_pairs/orquac/orquac_test.json
      --corpus_path $$AMLT_DATA_DIR/full_wiki_document.json
      --cache_dir $$AMLT_DATA_DIR/huggingface_cache/
      --trie_dict $$AMLT_DATA_DIR/trie_dict_t5-base_page_level.pkl
      --output_dir $$AMLT_OUTPUT_DIR/release_test/
      --learning_rate 1e-5
      --prepend_answers False
      --model_type t5-large
      --top_k 5
      --beam_size 5
``` 

We trained the models on 8*32GB NVIDIA V100 GPUs. 
We release our trained model checkpoints on the three datasets in this [link](https://drive.google.com/drive/folders/19ea3tuIFJkUYiwZ8eGMOTaJ0xnmydSDS?usp=sharing).


## Contact
If there is any problem, please email liyongqi0@gmail.com. Please do not hesitate to email me directly as I do not frequently check GitHub issues.
