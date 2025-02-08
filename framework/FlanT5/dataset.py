import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pdb
from tqdm import tqdm
import random
import copy
from multiprocessing import Process, Queue

############################
#  针对小数据集的数据集构造  #
############################

class DatasetIds(Dataset):
    def __init__(self, max_length, ids, mask=None, **kwargs):
        super().__init__()
        self.ids = ids
        self.mask = mask
        self.max_length = max_length


    def __len__(self):
        return len(self.ids) // self.max_length


    def __getitem__(self, index):
        bids = self.max_length * index
        eids = self.max_length * (index + 1)
        if not self.mask:
            return {
                "input_ids": torch.LongTensor(self.ids[bids: eids]),
                "labels": torch.LongTensor(self.ids[bids: eids])
            }
        else:
            return {
                "input_ids": torch.LongTensor(self.ids[bids: eids]),
                "labels": torch.LongTensor(self.ids[bids: eids]),
                "mask": torch.LongTensor(self.mask[bids: eids])
            }


class GPT2Dataset(Dataset):
    '''
        GPT2训练方法的数据集构造，没有Padding，通过EOS来截断。
    '''
    def __init__(self, tokenizer, datas, max_length):
        super().__init__()
        self.old_datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.index = 0
    
        if not self.tokenizer.bos_token:
            self.tokenizer.bos_token = "<s>"
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = "</s>"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
        self._preprocess()
        

    def _preprocess(self):
        self.datas = []
        for data in tqdm(self.old_datas):
            sample = ''
            for idx, item in enumerate(data):
                if idx == 0:
                    sample += item
                else:
                    sample += item[0] + item[1] 
            self.datas.append(sample)

        self.data_encoding = self.tokenizer(''.join([data + self.tokenizer.eos_token for data in tqdm(self.datas)]),return_tensors="pt")

    def __len__(self):
        return len(self.data_encoding['input_ids'][0]) // self.max_length

    def __getitem__(self, index):
        return self.data_encoding["input_ids"][0][index * self.max_length : (index + 1) * self.max_length], \
        self.data_encoding["input_ids"][0][index * self.max_length : (index + 1) * self.max_length], \
                self.data_encoding["attention_mask"][0][index * self.max_length : (index + 1) * self.max_length]

class BertDataset(Dataset):
    '''
        每个case之间存在空行，每个case的长度不超过max_length。
    '''
    def __init__(self, tokenizer, datas, max_length):
        super().__init__()
        self.old_datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
    
        if not self.tokenizer.bos_token:
            self.tokenizer.bos_token = "<s>"
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = "</s>"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._preprocess()
    
    def __len__(self):
        return len(self.datas)


    def _preprocess(self):
        self.datas = []
        for data in tqdm(self.old_datas):
            sample = ''
            for idx, item in enumerate(data):
                if idx == 0:
                    sample += item
                else:
                    sample += item[0] + item[1] 
            self.datas.append(sample)

    def __getitem__(self, index):
        data_encoding = self.tokenizer(
            self.datas[index] + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return data_encoding["input_ids"][0], data_encoding["input_ids"][0], data_encoding["attention_mask"][0]

class BertDataset_seq2seq(Dataset):
    '''
        每个case之间存在空行，每个case的长度不超过max_length。
    '''
    def __init__(self, tokenizer, datas, max_length):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
    
        if not self.tokenizer.bos_token:
            self.tokenizer.bos_token = "<s>"
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = "</s>"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item = self.datas[index]

        input_text, output = item[0], item[1]
        input_tokens = self.tokenizer(input_text, max_length=self.max_length,padding="max_length", truncation=True, return_tensors="pt")
        input_ids = input_tokens["input_ids"]
        attention_mask = input_tokens["attention_mask"]
        output_tokens = self.tokenizer(output, max_length=self.max_length // 4,padding="max_length", truncation=True, return_tensors="pt")
        labels = output_tokens["input_ids"]

        return input_ids[0], labels[0], attention_mask[0]

class BertDataset_seq2seq_ww(Dataset):
    '''
        每个case之间存在空行，每个case的长度不超过max_length。
    '''
    def __init__(self, tokenizer, datas, max_length):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
    
        if not self.tokenizer.bos_token:
            self.tokenizer.bos_token = "<s>"
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = "</s>"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        item = self.datas[index]

        input_text, output, weights = item[0], item[1], item[2]
        #print(item)
        input_tokens = self.tokenizer(input_text, max_length=self.max_length,padding="max_length", truncation=True, return_tensors="pt")
        input_ids = input_tokens["input_ids"]
        attention_mask = input_tokens["attention_mask"]
        output_tokens = self.tokenizer(output, max_length=self.max_length // 4,padding="max_length", truncation=True, return_tensors="pt")
        labels = output_tokens["input_ids"]

        return input_ids[0], labels[0], attention_mask[0], torch.full_like(labels[0], weights, dtype=torch.float16)