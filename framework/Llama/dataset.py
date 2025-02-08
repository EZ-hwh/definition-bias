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
        data = self.datas[index]

        for idx, item in enumerate(data):
            input_text, output = item[0], item[1]
            input_tokens = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = input_tokens["input_ids"]
            attention_mask = input_tokens["attention_mask"]
            output_tokens = self.tokenizer(output, max_length=self.max_length // 4,padding="max_length", truncation=True, return_tensors="pt")
            labels = output_tokens["input_ids"]

        return input_ids[0], labels[0], attention_mask[0]
        
class GPT2Dataset_onlyres(Dataset):
    '''
        GPT2训练方法的数据集构造，没有Padding，通过EOS来截断。只计算response的loss。
    '''
    def __init__(self, tokenizer, datas, max_length):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.index = 0
        self.seed = 42
        
        if not self.tokenizer.bos_token:
            self.tokenizer.bos_token = "<s>"
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = "</s>"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._preprocess()

    def _preprocess(self):
        self.input_ids = []
        self.lengths = []
        self.labels = []

        self.batch_input_ids = []
        self.batch_labels = []
        self.batch_attention_mask = []
        numseqs = 0
        
        for data in tqdm(self.datas):
            sample_input_ids = []
            sample_labels = []
            sample_attention_mask = []

            for idx, item in enumerate(data):
                inputs, output = ' '+item[0], item[1]

                input_tokens = self.tokenizer(inputs, padding=False, truncation=False, add_special_tokens=True)
                # 不能取后面的，否则[Round]就不在了
                input_tokens = input_tokens["input_ids"]

                len_input = len(input_tokens)
                output_tokens = self.tokenizer(output, padding=False, truncation=False, add_special_tokens=False)
                output_tokens = output_tokens["input_ids"]

                sample_input_ids += input_tokens + output_tokens + [self.tokenizer.eos_token_id]
                sample_labels += [-100] * len_input + output_tokens + [self.tokenizer.eos_token_id]
                sample_attention_mask += [1] * (len(input_tokens) + len(output_tokens) + 1)
            
            sample_length = len(sample_input_ids)

            if sample_length != 0 and sample_length < self.max_length:
                if len(self.batch_input_ids) == 0 or numseqs + sample_length > self.max_length:
                    self.batch_input_ids.append(sample_input_ids)
                    self.batch_labels.append(sample_labels)
                    self.batch_attention_mask.append(sample_attention_mask)
                    numseqs = sample_length
                else: #if numseqs + sample_length <= self.max_length:
                    self.batch_input_ids[-1].extend(sample_input_ids)
                    self.batch_labels[-1].extend(sample_labels)
                    self.batch_attention_mask[-1].extend(sample_attention_mask)
                    numseqs += sample_length

                # self.input_ids += sample_input_ids
                # self.labels += sample_labels
            
                # self.input_ids += [self.tokenizer.eos_token_id]
                # self.labels += [self.tokenizer.eos_token_id]
            #self.lengths.append(len(sample_input_ids))

        #self.attention_mask = [1] * len(self.input_ids)
        for idx in range(len(self.batch_input_ids)):
            self.batch_input_ids[idx] += [self.tokenizer.pad_token_id] * (self.max_length - len(self.batch_input_ids[idx]))
            self.batch_labels[idx] += [-100] * (self.max_length - len(self.batch_labels[idx]))
            self.batch_attention_mask[idx] += [0] * (self.max_length - len(self.batch_attention_mask[idx]))

    def __len__(self):
        return len(self.batch_input_ids)

    def __getitem__(self, index):
        
        return  torch.tensor(self.batch_input_ids[index]), \
                torch.tensor(self.batch_labels[index]), \
                torch.tensor(self.batch_attention_mask[index])


class BertDataset_onlyres(Dataset):
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

        # meta_prompt = self.datas[0][0]
        # meta_tokens = self.tokenizer(meta_prompt, padding=False, truncation=False, add_special_tokens=False)
        # meta_tokens = meta_tokens["input_ids"][:]
        
        while True:
            data = self.datas[index]
            # sample_input_ids = copy.copy(meta_tokens)
            # sample_labels = [-100] * len(sample_input_ids)
            sample_input_ids = []
            sample_labels = []

            for idx, item in enumerate(data):
                # pdb.set_trace()
                input_text, output = item[0], item[1]
                # 防止梯度消失
                input_tokens = self.tokenizer(input_text, padding=False, truncation=False, add_special_tokens=True)
                # 不能取后面的，否则[Round]就不在了
                input_tokens = input_tokens["input_ids"][:]

                len_input = len(input_tokens)
                output_tokens = self.tokenizer(output, padding=False, truncation=False, add_special_tokens=False)
                output_tokens = output_tokens["input_ids"][:]

                sample_input_ids += input_tokens + output_tokens
                sample_labels += [-100] * len_input + output_tokens
                # pdb.set_trace()

            sample_input_ids += [self.tokenizer.eos_token_id]
            sample_labels += [self.tokenizer.eos_token_id]
            sample_attention_mask = [1] * len(sample_input_ids)
            
            sample_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(sample_input_ids))
            sample_labels += [-100] * (self.max_length - len(sample_labels))
            sample_attention_mask += [0] * (self.max_length - len(sample_attention_mask))
            
            sample_input_ids = sample_input_ids[:self.max_length]
            sample_labels = sample_labels[:self.max_length]
            sample_attention_mask = sample_attention_mask[:self.max_length]
            if sample_labels == [-100] * self.max_length:
                index = random.randint(0, len(self.datas) - 1)
            else:
                break
                #print(data)

        return torch.tensor(sample_input_ids), torch.tensor(sample_labels), torch.tensor(sample_attention_mask)


if __name__ == '__main__':
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('../../../models/llama-2-13b-hf')
    datas = [[('This is a test case1', 'This is output1')], [('This is a test case2', 'This is output2')]]
    dataset = GPT2Dataset_onlyres(tokenizer, datas, 512)
    for data in dataset:
        print(data)
        print(tokenizer.decode(data[0]))