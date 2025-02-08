import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset
from itertools import permutations
from collections import defaultdict
from transformers import BertTokenizerFast, RobertaTokenizer, AutoTokenizer

class NERDataset(Dataset):
    '''
    dataset for NER
    '''
    def __init__(self, data_path, tokenizer, labels):
        self._load_dataset(data_path)
        self.tokenizer = tokenizer
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}

    def _load_dataset(self, data_path):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def _find_pos(self, entity, input_ids):
        entity_tokens = self.tokenizer.tokenize(entity)
        entity_ids = self.tokenizer.convert_tokens_to_ids(entity_tokens)
        ret = []
        for index in range(len(input_ids)):
            if entity_ids[0] == input_ids[index]:
                flag = True
                for i in range(len(entity_ids)):
                    if index + i >= len(input_ids) or input_ids[index+i] != entity_ids[i]:
                        flag=False
                        break
                if flag:
                    ret.append((index,index+len(entity_ids)))
        return ret

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        output = self.tokenizer(data['sentence'], return_offsets_mapping=True, return_token_type_ids=True, max_length=512)
        input_ids = output['input_ids']
        token_type_ids = output['token_type_ids']
        labels = torch.zeros(len(self.labels),len(input_ids), len(input_ids)).int()
        offset_mapping = output['offset_mapping']
        for entity in data['entities']:
            if entity['type'] not in self.labels:
                continue
            entity_spans = self._find_pos(entity['name'], input_ids)
            for s,e in entity_spans:
                labels[self.label2id[entity['type']]][s][e-1] = 1
        return torch.IntTensor(input_ids), torch.IntTensor(token_type_ids), labels, offset_mapping, data['sentence']

def collate_fn_cuda(batch):
    bz = len(batch)
    cls_num = batch[0][2].shape[0]
    maxlen = max([len(item[0]) for item in batch])
    batch_input_ids = []
    batch_token_type_ids = []
    batch_labels = torch.zeros(bz,cls_num,maxlen,maxlen)

    for index, (input_ids, token_type_ids, labels, _, _) in enumerate(batch):
        batch_input_ids.append(input_ids)
        batch_token_type_ids.append(token_type_ids)
        seqlen = len(input_ids)
        batch_labels[index][:,:seqlen,:seqlen] = labels

    batch_input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True)
    batch_token_type_ids = torch.nn.utils.rnn.pad_sequence(batch_token_type_ids, batch_first=True)

    return [
        batch_input_ids.cuda(),
        batch_token_type_ids.cuda(),
        batch_labels.cuda()
    ]


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    dataset = NERDataset('/mnt/huangwenhao/data122/datasets/Information Extraction/academic_dataset/IE_INSTRUCTIONS/NER/ACE 2004/train.json', tokenizer, ['person', 'location', 'organization'])
    for i in dataset:
        print(i)