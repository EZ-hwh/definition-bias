import os
import csv
import json
import argparse
import pandas as pd
import pdb
import sys
import time


import torch
import deepspeed
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelWithLMHead, T5Tokenizer, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import ijson
import pickle
from peft import LoraConfig, PeftModel, get_peft_model

from dataset import GPT2Dataset, BertDataset, GPT2Dataset_onlyres, BertDataset_onlyres
import random

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    fan_in_fan_out=False,
    task_type="CAUSAL_LM",
)

def get_data(args, rank):
    
    # origin_dataloader = WholeLoader(args.data_path, args.train_config)
    # origin_dataset = origin_dataloader.get_dataset(use_nickname=args.use_nickname)
    origin_dataset = []
    with open(args.data_path) as f:
        for line in f.readlines():
            origin_dataset.append(json.loads(line))

    datas = []
    if rank == 0:
        print('Origin data format:')
        print(json.dumps(origin_dataset[:5],ensure_ascii=False,indent=4))
        pbar = tqdm(total=len(origin_dataset), mininterval=0)
        print('='*50)

    for idx in range(len(origin_dataset)):
        datas.append([(' '+origin_dataset[idx]['items'][0]['content'], origin_dataset[idx]['items'][1]['content'])])
        if rank == 0:
            pbar.update(1)

    if rank == 0:
        print('Handled data format:')
        print(datas[0])
        print(len(datas))
        print('='*50)

    if args.max_training_samples == -1:
        return datas
    return datas[:max_training_samples]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
    parser.add_argument("--max_epoches",type=int,default=5,help="max epoches to run dataloader")
    parser.add_argument("--max_training_samples",type=int,default=-1,help="max number of training samples")
    parser.add_argument("--data_path",type=str,default='../weighted_dataset/516v1/ift_data.pkl',help="the floader to load training data")
    parser.add_argument("--model_path",type=str,default='/data/qyh/instruction_tuning/llama/ckp/add_ft_llama_xyp_517_v1/add_ft_llama_xyp_517_v1_epoch2',help="the floader to load model")
    parser.add_argument("--ds_config_path", type=str, default='configs/ds_config_llama_lora_13B.json', help="the config file of deepspeed settings")

    parser.add_argument("--max_length",type=int,default=1024,help="max token length")
    parser.add_argument("--load_lora",action="store_true",help="whether load ckpts")
    parser.add_argument("--load_lora_path",type=str,default="",help="the floader to load ckpts(.pt)")
    parser.add_argument("--save_dir",type=str,default="ckp/",help="the floader to save ckpts(.pt)")
    parser.add_argument("--save_name",type=str,default="bloom_new",help="the floader extension name")
    parser.add_argument("--save_steps",type=int,default=1000,help="how many step to save a model")
    parser.add_argument("--overwrite_data",action="store_true",help="how many step to save a model")
    parser.add_argument("--dataset_type",choices=['GPT2Dataset','BertDataset','GPT2Dataset_onlyres','BertDataset_onlyres'],help="The type of dataset for dataloader")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    with open(args.ds_config_path) as f:
        DS_CONFIG = json.load(f)

    os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)

    device = torch.device("cuda")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    
    deepspeed.init_distributed()
    model_name = args.model_path
    print('model_name:',model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    # FIXME: add special tokens for cutegpt multi-round generation
    # st = ["<end>"]
    # tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + st})
    # print(tokenizer.additional_special_tokens)

    print(tokenizer.additional_special_tokens)
    random.seed(10)

    print('data_path:', args.data_path)
    datas = get_data(args, torch.distributed.get_rank())
    # pdb.set_trace()
    train_dataset = eval(f"{args.dataset_type}")(
        tokenizer,
        datas, # your data preprocessing function
        args.max_length # your max input length
    )
    
    print('dataset loaded!')
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        sampler=train_sampler,
        batch_size=DS_CONFIG["train_micro_batch_size_per_gpu"]
    )

    print('add_special_token....')
    model = LlamaForCausalLM.from_pretrained(model_name, use_flash_attention_2=True)


    load_lora = args.load_lora
    if load_lora:
        # 如果load参数
        print('parameter loaded!')
        print(args.load_lora_path)
        model = PeftModel.from_pretrained(model, args.load_lora_path, is_trainable=True)
        time.sleep(torch.distributed.get_rank() * 20)
    else:
        # 如果重新训练
        print('training from scratch')
        model = get_peft_model(model, lora_config)

    if torch.distributed.get_rank() == 0:
        model.print_trainable_parameters()
        print('load peft model')

    # model.resize_token_embeddings(len(tokenizer))

    engine, _, _, _ = deepspeed.initialize(
        config=DS_CONFIG,
        model=model, 
        model_parameters=model.parameters()
    )
    print("model loaded.")

    args.max_steps = args.max_epoches * len(train_dataloader)

    global_step = 0
    engine.train()
    for epoch in range(args.max_epoches):
        print(epoch)

        losses = []
        if torch.distributed.get_rank() != -1:
            train_sampler.set_epoch(epoch)
        if torch.distributed.get_rank() == 0:
            pbar = tqdm(range(len(train_dataloader)))

        for batch in train_dataloader:
            loss = engine(
                input_ids = batch[0].to(device),
                labels = batch[1].to(device),
                attention_mask = batch[2].to(device),
            ).loss
            engine.backward(loss)
            engine.step()

            global_step += 1
            losses.append(loss.item())

            if torch.distributed.get_rank() == 0:
                pbar.update()
                pbar.set_description(f"loss: {sum(losses[-200: ]) / len(losses[-200: ])}")

            if global_step >= args.max_steps:
                break
        
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            engine.save_pretrained(os.path.join(args.save_dir, args.save_name, f'ep_{epoch}'))
        dist.barrier()

        if torch.distributed.get_rank() == 0:
            pbar.close()
        if global_step >= args.max_steps:
            break