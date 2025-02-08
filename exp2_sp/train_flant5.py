import os
import csv
import json
import argparse
import pdb
import sys
import time


import torch
import deepspeed
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ijson
from data.data_utils import WholeLoader
import pickle
from peft import LoraConfig, PeftModel, get_peft_model
from dataset import BertDataset_seq2seq
import random
from configs.lora_config import lora_config

template_wo_example = 'Instruction:{0}\n\nInput:{1}\nOutput:'

def get_data(args, rank):
    
    origin_dataloader = WholeLoader(args.data_path, args.train_config)
    origin_dataset = origin_dataloader.get_dataset(use_nickname=args.use_nickname)
    datas = []
    if rank == 0:
        if args.use_nickname:
            print("Mixed nickname in training.")
        print('Origin data format:')
        print(json.dumps(origin_dataset[:5],ensure_ascii=False,indent=4))
        pbar = tqdm(total=len(origin_dataset), mininterval=0)
        print('='*50)

    for idx in range(len(origin_dataset)):
        datas.append([(template_wo_example.format(origin_dataset[idx]['instruction'], origin_dataset[idx]['input']), origin_dataset[idx]['output'])])

        if rank == 0:
            pbar.update(1)

    if rank == 0:
        print('Handled data format:')
        print(datas[0])
        print(len(datas))
        print('='*50)

    if args.max_training_samples == -1:
        return datas
    return datas[:args.max_training_samples]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
    parser.add_argument("--max_epoches",type=int,default=5,help="max epoches to run dataloader")
    parser.add_argument("--max_training_samples",type=int,default=-1,help="max number of training samples")
    parser.add_argument("--data_path",type=str,default='../weighted_dataset/516v1/ift_data.pkl',help="the floader to load training data")
    parser.add_argument("--model_path",type=str,default='/data/qyh/instruction_tuning/llama/ckp/add_ft_llama_xyp_517_v1/add_ft_llama_xyp_517_v1_epoch2',help="the floader to load model")
    parser.add_argument("--ds_config_path", type=str, default='configs/ds_config_llama_lora_13B.json', help="the config file of deepspeed settings")
    parser.add_argument("--train_config", type=str, help="training config")
    parser.add_argument("--use_nickname", action="store_true", help="Whether to use nickname in dataset name")

    parser.add_argument("--max_length",type=int,default=1024,help="max token length")
    parser.add_argument("--flashattn",action="store_true",help='Whether to use flash attention')
    parser.add_argument("--load_lora",action="store_true",help="whether load ckpts")
    parser.add_argument("--use_lora",action="store_true",help="Whether to use LoRA, the default is to perform fully finetuned.")
    parser.add_argument("--load_lora_path",type=str,default="",help="the floader to load ckpts(.pt)")
    parser.add_argument("--save_dir",type=str,default="ckp/",help="the floader to save ckpts(.pt)")
    parser.add_argument("--save_name",type=str,default="bloom_new",help="the floader extension name")
    parser.add_argument("--save_steps",type=int,default=1000,help="how many step to save a model")
    parser.add_argument("--dataset_type",choices=['GPT2Dataset','BertDataset','GPT2Dataset_onlyres','BertDataset_onlyres'],help="The type of dataset for dataloader")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    with open(args.ds_config_path) as f:
        DS_CONFIG = json.load(f)

    if args.local_rank == 0:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    device = torch.device("cuda")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    
    if torch.distributed.get_rank() == 0:
        print(DS_CONFIG)

    deepspeed.init_distributed()
    model_name = args.model_path
    print('model_name:',model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print('data_path:', args.data_path)

    datas = get_data(args, torch.distributed.get_rank())
    
    train_dataset = BertDataset_seq2seq(
        tokenizer,
        datas, # your data preprocessing function
        args.max_length # your max input length
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        sampler=train_sampler,
        batch_size=DS_CONFIG["train_micro_batch_size_per_gpu"]
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, )
    
    if not args.use_lora:
        if torch.distributed.get_rank() == 0:
            print('Fully finetuned...')
    else:
        if args.load_lora:
            if torch.distributed.get_rank() == 0:
                print('LoRA Parameter loaded!')
                print(args.load_lora_path)
            model = PeftModel.from_pretrained(model, args.load_lora_path, is_trainable=True)
        else:
            if torch.distributed.get_rank() == 0:
                print('Training from scratch')
            model = get_peft_model(model, lora_config)

        if torch.distributed.get_rank() == 0:
            model.print_trainable_parameters()

    engine, _, _, _ = deepspeed.initialize(
        config=DS_CONFIG,
        model=model, 
        model_parameters=model.parameters()
    )

    args.max_steps = args.max_epoches * len(train_dataloader)

    global_step = 0
    engine.train()
    for epoch in range(args.max_epoches):
        losses = []
        if torch.distributed.get_rank() != -1:
            train_sampler.set_epoch(epoch)
        if torch.distributed.get_rank() == 0:
            pbar = tqdm(range(len(train_dataloader)))

        for batch in train_dataloader:
            loss = engine(
                input_ids = batch[0].to(device),
                labels = batch[1].to(device),
                attention_mask = batch[2].to(device)
            ).loss
            engine.backward(loss)
            engine.step()

            global_step += 1
            losses.append(loss.item())

            if global_step % args.save_steps == 0:
                if args.use_lora:
                    dist.barrier()
                    if torch.distributed.get_rank() == 0:
                        engine.save_pretrained(f"{args.save_dir}/{args.save_name}_{global_step}")
                    dist.barrier()
                else:
                    engine.save_16bit_model(f'{args.save_dir}/{args.save_name}_{global_step}')
                    os.makedirs(f'{args.save_dir}/{args.save_name}_{global_step}', exist_ok=True)
                    with open(f'{args.save_dir}/{args.save_name}_{global_step}/config.json','w') as f:
                        json.dump(engine.module.config.to_dict(), f)
                    tokenizer.save_pretrained(f'{args.save_dir}/{args.save_name}_{global_step}')

            if torch.distributed.get_rank() == 0:
                pbar.update()
                pbar.set_description(f"loss: {sum(losses[-200: ]) / len(losses[-200: ])}")
            if global_step >= args.max_steps:
                break
        
        
        if args.use_lora:
            dist.barrier()
            if torch.distributed.get_rank() == 0:
                engine.save_pretrained(f"{args.save_dir}/{args.save_name}_epoch{epoch}")
            dist.barrier()
        else:
            engine.save_16bit_model(f'{args.save_dir}/{args.save_name}_epoch{epoch}')
            os.makedirs(f'{args.save_dir}/{args.save_name}_epoch{epoch}', exist_ok=True)
            with open(f'{args.save_dir}/{args.save_name}_epoch{epoch}/config.json','w') as f:
                json.dump(engine.module.config.to_dict(), f)
            tokenizer.save_pretrained(f'{args.save_dir}/{args.save_name}_epoch{epoch}')

        if torch.distributed.get_rank() == 0:
            pbar.close()
        if global_step >= args.max_steps:
            break
