# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import os


import argparse
import logging
import os
import random
import glob
import timeit
import json


from tqdm import tqdm, trange
from copy import copy
import re
import torch
import copy
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
import transformers
from transformers import T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from retriever_utils import FinetuningDataset
from modeling import Generative_Retrieval

import pickle
from torch.cuda.amp import autocast as autocast
import numpy as np
from datasets import load_dataset
from multiprocessing import Pool
from utils import Trie

from contextlib import contextmanager
# In[2]:
np_str_obj_array_pattern = re.compile(r'[SaUO]')

logger = logging.getLogger(__name__)


transformers.logging.set_verbosity_error()


# In[3]:


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)




def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#######################################################yongqi
def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

def prefix_allowed_tokens_fn(batch_id, sent):
    return decoder_trie.get(sent.tolist())

def dist_gather_tensor(t):
    if t is None:
        return None
    t = t.contiguous()

    all_tensors = [torch.empty_like(t) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_tensors, t)

    all_tensors[torch.distributed.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors



def train(args, model, tokenizer):
    DatasetClass = FinetuningDataset
    train_dataset = DatasetClass(args.train_file, tokenizer,
                                 args.load_small,
                                 query_max_seq_length=args.query_max_seq_length,target_max_seq_length=args.target_max_seq_length, prepend_answers=args.prepend_answers)

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))


    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

 

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)


    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


   # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps == 0:
        args.warmup_steps = int(t_total * args.warmup_portion)
        
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)

    global_step_list = []
    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)
     
#######################################################yongqi
        for step, batch in enumerate(epoch_iterator):




#######################################################yongqi
            model.train()
            query_input_ids = batch['query_input_ids']
            query_attention_mask = batch['query_attention_mask']

            target_input_ids = batch['target_input_ids']
            target_attention_mask = batch['target_attention_mask']

            target_input_ids[target_attention_mask == 0] = -100


            inputs = {'args': args,
                      'query_input_ids': query_input_ids.to(args.device),
                      'query_attention_mask': query_attention_mask.to(args.device),
                      'target_input_ids': target_input_ids.to(args.device),
                      'target_attention_mask': target_attention_mask.to(args.device),
                      'mode': "train"}
            if args.fp16:
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    loss = model(**inputs)
            else:
                loss = model(**inputs)


            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    # Updates the scale for next iteration.
                    scaler.update()

                    scheduler.step()
                    model.zero_grad()
                    global_step += 1
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                # print('loss', loss.item())
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss


        if args.save_steps == -1:
            global_step_list.append(global_step)
            if args.local_rank in [-1, 0]:

                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(
                    model, 'module') else model
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, 'checkpoint-{}'.format(global_step))
                torch.save(model_to_save.state_dict(), output_dir+"model.pt")
                logger.info("Saving model checkpoint to %s", output_dir)


    return global_step, tr_loss / global_step, global_step_list

# In[5]:
def evaluate_dev(args, model, tokenizer):
    args.eval_batch_size = args.per_gpu_eval_batch_size
    # eval dataset load here to avoid load every time
    DatasetClass = FinetuningDataset
    eva_dataset = DatasetClass(args.dev_file, tokenizer,
                                 args.load_small,
                                 query_max_seq_length=args.query_max_seq_length, target_max_seq_length=args.target_max_seq_length, prepend_answers=args.prepend_answers)
    eval_sampler = RandomSampler(
        eva_dataset) if args.local_rank == -1 else DistributedSampler(eva_dataset)

    eval_dataloader = DataLoader(
        eva_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)


    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Eval!
    logger.info("***** Running evaluation dev *****")
    logger.info("  Num examples = %d", len(eva_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    correct_num = 0.0
    total_num = 0.0


    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):

        model.eval()
        query_input_ids = batch['query_input_ids']
        query_attention_mask = batch['query_attention_mask']

        target_input_ids = batch['target_input_ids']
        target_attention_mask = batch['target_attention_mask']

        query_text = batch['query_text']
        target_text = batch['target_text']

        with torch.no_grad():
            inputs = {'args': args,
                      'query_input_ids': query_input_ids.to(args.device),
                      'query_attention_mask': query_attention_mask.to(args.device),
                      'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn,
                      'mode': "dev"}
            outputs = model(**inputs)

        predicted_target_text = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]

        for i in range(len(query_text)):
            total_num = total_num + 1
            if target_text[i] == predicted_target_text[i]:
                correct_num = correct_num + 1
    correct_num_gather =  torch.from_numpy(np.array([correct_num])).to(args.device)
    total_num_gather =  torch.from_numpy(np.array([total_num])).to(args.device)

    if args.local_rank != -1:
        correct_num_gather = torch.sum(dist_gather_tensor(correct_num_gather))
        total_num_gather = torch.sum(dist_gather_tensor(total_num_gather))
        correct_num = correct_num_gather.item()
        total_num = total_num_gather.item()
    acc = correct_num/total_num
    logger.info("  correct_num = %d", correct_num)
    logger.info("  total_num = %d", total_num)
    logger.info("  acc = %s", str(acc))
    return acc

def evaluate_test(args, model, tokenizer, eva_dataset, eval_dataloader):


    # Eval!
    logger.info("***** Running evaluation dev *****")
    logger.info("  Num examples = %d", len(eva_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    correct_num = [0.0]*args.top_k
    total_num = 0.0
    output_dict = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):

        model.eval()
        query_input_ids = batch['query_input_ids']
        query_attention_mask = batch['query_attention_mask']

        target_input_ids = batch['target_input_ids']
        target_attention_mask = batch['target_attention_mask']

        query_text = batch['query_text']
        target_text = batch['target_text']

        
        with torch.no_grad():
            inputs = {'args': args,
                      'query_input_ids': query_input_ids.to(args.device),
                      'query_attention_mask': query_attention_mask.to(args.device),
                      'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn,
                      'mode': "test"}
            outputs = model(**inputs)



        predicted_target_text = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]

        for i in range(len(query_text)):
            total_num = total_num + 1
            entry={}
            entry['question'] = query_text[i]
            entry['answers'] = [batch['answer_text'][i]]
            entry['ctxs'] = []

            for j in range(args.top_k):
                if target_text[i] == predicted_target_text[i*args.top_k+j]:
                    correct_num[j] = correct_num[j]  + 1
                if predicted_target_text[i*args.top_k+j] in title2idx:
                    idx = title2idx[predicted_target_text[i*args.top_k+j]]
                    entry['ctxs'].append(passage_corpus[idx])
                else:
                    entry['ctxs'].append({"id": 0, 'title':predicted_target_text[i*args.top_k+j], 'text': ""})
            output_dict.append(entry)
    for k in [1,3,5,10,20,50,100]:
        new_correct_num = correct_num[:k]
        correct_num_k = sum(new_correct_num)
        recall_k = correct_num_k/total_num


        mrr = 0.0
        for j in range(len(new_correct_num)):
            mrr += float(new_correct_num[j])/(j+1)
        mrr = mrr/total_num
        logger.info("correct_num = %s", correct_num_k)
        logger.info("total_num = %s", total_num)
        logger.info("recall @ " + str(k) + " = %s", str(recall_k))
        logger.info("mrr @ " + str(k) + " = %s", str(mrr))
    return output_dict
    
def dist_gather_tensor(t):
    if t is None:
        return None
    t = t.contiguous()

    all_tensors = [torch.empty_like(t) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_tensors, t)

    all_tensors[torch.distributed.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)

    return all_tensors

parser = argparse.ArgumentParser()

# data
parser.add_argument("--train_file", default="/home/v-yongqili/project/GCoQA/data/QA_pairs/topiocqa/topiocqa_train.json",
                    type=str, required=False,
                    help="training file ")
parser.add_argument("--dev_file", default="/home/v-yongqili/project/GCoQA/data/QA_pairs/topiocqa/topiocqa_dev.json",
                    type=str, required=False,
                    help="dev_file ")
parser.add_argument("--test_file", default="/home/v-yongqili/project/GCoQA/data/QA_pairs/topiocqa/topiocqa_test.json",
                    type=str, required=False,
                    help="test_file ")
parser.add_argument("--corpus_path", default="/home/v-yongqili/project/GCoQA/data/full_wiki_segments.json",
                    type=str, required=False,
                    help="dev_file ")
parser.add_argument("--trie_dict", default="/home/v-yongqili/project/GCoQA/data/trie_dict_t5-base_section_level.pkl",
                    type=str, required=False,
                    help="dev_file ")
parser.add_argument("--cache_dir", default="/home/v-yongqili/project/GCoQA/data/huggingface_cache/", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--output_dir", default='./release_test1', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")

parser.add_argument("--pretrained_ckpt_path", default=None, type=str, required=False,
                    help="pretrained_passage_encoder_paramaters")
parser.add_argument("--test_ckpt_path", default=None, type=str, required=False,
                    help="trained_dul_encoder_paramaters")

parser.add_argument("--load_small", default=False, type=str2bool, required=False,
                    help="whether to load just a small portion of data during development")
parser.add_argument("--num_workers", default=4, type=int, required=False,
                    help="number of workers for dataloader")

# training
parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_test", default=True, type=str2bool,
                    help="Whether to run eval on the test set.")
# parameters
parser.add_argument("--prepend_answers", default=False, type=str2bool,
                    help="Whether to prepend answers.")


parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--per_gpu_test_batch_size", default=2, type=int,
                    help="Batch size per GPU/CPU for evaluation.")

parser.add_argument("--learning_rate", default=1e-4, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=40, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument('--do_lower_case', type=str2bool, default=True,
                    help="tokenizer do_lower_case")

parser.add_argument('--top_k', type=int, default=5,
                    help="the number of retrieved passages")
parser.add_argument('--beam_size', type=int, default=5,
                    help="the number of retrieved passages")

parser.add_argument('--query_max_seq_length', type=int, default=384,
                    help="passage_max_seq_length")
parser.add_argument('--target_max_seq_length', type=int, default=64,
                    help="passage_max_seq_length")

parser.add_argument('--logging_steps', type=int, default=10,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=-1,
                    help="Save checkpoint every X updates steps.")

parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', default=True, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', default=False, type=str2bool,
                    help="Whether to use 16-bit (mixed) precision")


parser.add_argument("--model_type", default="t5-base",
                    type=str, required=False,
                    help="the type of pretrining model ")

args, unknown = parser.parse_known_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
               args.local_rank, device, bool(args.local_rank != -1), args.fp16)



# Set seed
set_seed(args)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()


tokenizer = T5Tokenizer.from_pretrained(args.model_type,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir)


with open(args.trie_dict, 'rb') as f:
    decoder_trie = Trie.load_from_dict(pickle.load(f))
logger.info("decoder_trie len %s", decoder_trie.len)

model = Generative_Retrieval(args)

if args.pretrained_ckpt_path is not None:
    model.load_state_dict(torch.load(args.pretrained_ckpt_path, map_location=torch.device('cpu')))
    logger.info("load checkpoint from %s", args.pretrained_ckpt_path)




passage_corpus = load_dataset('json', data_files=args.corpus_path, split="train")
logger.info("passage_corpus info %s", passage_corpus)


title2idx = {}
with open(args.corpus_path, 'r') as f:
    num = 0
    for line in tqdm(f):
        line = json.loads(line)

        title2idx[line['title']] = num
        num+=1
print('len title2idx', len(title2idx))



if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

model.to(args.device)





logger.info("Training/evaluation parameters %s", args)
if args.do_train:
    global_step, tr_loss, global_step_list = train(
        args, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)

    if args.local_rank != -1:
        torch.distributed.barrier()

    # do eval
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    max_acc = -0.1
    best_global_step = 0

    for global_step in global_step_list:

        model = Generative_Retrieval(args)
        args.test_ckpt_path = os.path.join(
            args.output_dir, 'checkpoint-{}'.format(global_step))+"model.pt"

        model.load_state_dict(torch.load(args.test_ckpt_path, map_location=torch.device('cpu')))

        model.to(args.device)


        logger.info("test the checkpoint from %s", args.test_ckpt_path)
        acc = evaluate_dev(args, model, tokenizer)
        if args.local_rank in [-1, 0]:
            tb_writer.add_scalar('acc_dev', acc, global_step)
        if acc > max_acc:
            max_acc = acc
            best_global_step = global_step
        logger.info("max_acc = %s", str(max_acc))
        logger.info("best_global_step = %s", str(best_global_step))


if args.do_test and args.local_rank in [-1, 0]:
    model = Generative_Retrieval(args)
    if args.do_train:
        args.test_ckpt_path = os.path.join(
            args.output_dir, 'checkpoint-{}'.format(best_global_step))+"model.pt"
        model.load_state_dict(torch.load(args.test_ckpt_path, map_location=torch.device('cpu')))
        model.to(args.device)
    else:
        model.load_state_dict(torch.load(args.test_ckpt_path, map_location=torch.device('cpu')))
        model.to(args.device)   
    logger.info("test the checkpoint from %s", args.test_ckpt_path)


    args.eval_batch_size = args.per_gpu_test_batch_size
    # eval dataset load here to avoid load every time
    DatasetClass = FinetuningDataset
    eva_dataset = DatasetClass(args.test_file, tokenizer,
                                 args.load_small,
                                 query_max_seq_length=args.query_max_seq_length, target_max_seq_length=args.target_max_seq_length, prepend_answers=args.prepend_answers)
    eval_sampler = SequentialSampler(eva_dataset)

    eval_dataloader = DataLoader(
        eva_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)


    # Evaluate on test set
    output_dict = evaluate_test(args, model, tokenizer, eva_dataset, eval_dataloader)
    output_dict_file = os.path.join(
        args.output_dir, 'output_dict_file-test.json')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(output_dict_file, 'w') as json_file:
          json.dump(output_dict, json_file)


    # eva_dataset = DatasetClass(args.dev_file, tokenizer,
    #                              args.load_small,
    #                              query_max_seq_length=args.query_max_seq_length, target_max_seq_length=args.target_max_seq_length, prepend_answers=args.prepend_answers)
    # eval_sampler = SequentialSampler(eva_dataset)

    # eval_dataloader = DataLoader(
    #     eva_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)


    # # Evaluate on dev set
    # output_dict = evaluate_test(args, model, tokenizer, eva_dataset, eval_dataloader)
    # output_dict_file = os.path.join(
    #     args.output_dir, 'output_dict_file-dev.json')
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # with open(output_dict_file, 'w') as json_file:
    #       json.dump(output_dict, json_file)


    # eva_dataset = DatasetClass(args.train_file, tokenizer,
    #                              args.load_small,
    #                              query_max_seq_length=args.query_max_seq_length, target_max_seq_length=args.target_max_seq_length, prepend_answers=args.prepend_answers)
    # eval_sampler = SequentialSampler(eva_dataset)

    # eval_dataloader = DataLoader(
    #     eva_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)


    # # Evaluate on dev set
    # output_dict = evaluate_test(args, model, tokenizer, eva_dataset, eval_dataloader)
    # output_dict_file = os.path.join(
    #     args.output_dir, 'output_dict_file-train.json')
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # with open(output_dict_file, 'w') as json_file:
    #       json.dump(output_dict, json_file)