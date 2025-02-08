import os, sys, time, ljqpy, math, re, json
import unicodedata
import torch
import torch.nn as nn
import torch.functional as F
from tqdm import tqdm
import numpy as np
from functools import partial
from collections import defaultdict
import argparse
from config import config
import random
import copy
from utils import TN, restore_token_list, GetTopSpans, FindValuePos
from transformers import BertTokenizer, BertModel, set_seed, BertConfig

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_dp", type=str, help='dataset for training the parameters')
parser.add_argument("--test_dp", type=str, help='dataset for testing the parameters')
parser.add_argument("--refer_weight_file", type=str, help='weight of reference model')
parser.add_argument("--test_weight_file", type=str, help='weight of RC and EE model')
parser.add_argument("--model_path", type=str, help='the path to the pretrained model')
parser.add_argument("--test_file", type=str, help='the file to be tested')
parser.add_argument("--result_path", type=str, help='the path to store the result_path')
parser.add_argument("--maxlen", type=int)
args = parser.parse_args()

# Params
rc_threshold = 0.5
ee_threshold = 0.5

## Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, maxlen):
        global rel2id
        self.items = []
        self.maxlen = maxlen
        for z in data:
            item = {}
            item['tid'] = torch.tensor(tokenizer.encode(z['sentence'])[:maxlen])
            self.items.append(item)
    def __len__(self): return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k]
        return item['tid']

class PU_mid_loss(nn.Module):
    def __init__(self, mid=0, pi=0.1):
        super().__init__()
        self.mid = mid
        self.pi = pi

    def forward(self,y_true,y_pred):
        eps = torch.tensor(1e-6).cuda()
        y_true = y_true.double()
        pos = torch.sum(y_true * y_pred, 1) / torch.maximum(eps, torch.sum(y_true, 1))
        pos = - torch.log(pos + eps)
        neg = torch.sum((1-y_true) * y_pred, 1) / torch.maximum(eps, torch.sum(1-y_true, 1))
        neg = torch.abs(neg - self.mid) 
        neg = - torch.log(1 - neg + eps)
        return torch.mean(self.pi*pos + neg)

class DatasetEE(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.items = []
        for i, z in enumerate(data):
            text, spo_list = z['sentence'], z['relations']
            spo_list = [{
                's': item['head']['name'],
                'p': item['type'],
                'o': item['tail']['name']
            } for item in spo_list]
            labels = z.get('rc_pred', list(set(x['p'] for x in spo_list)))  
            tokens = tokenizer.tokenize(text)
            otokens = restore_token_list(text, tokens)
            tid = tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.sep_token_id]
            for label in labels:
                prompt = tokenizer.encode(label)
                plen = len(prompt)
                item = {'text':text, 'spo_list':spo_list}
                item['id'] = i
                item['plen'] = plen
                item['otokens'] = otokens
                item['label'] = label
                item['tid'] = torch.tensor((prompt + tid)[:maxlen])
                self.items.append(item)
    def __len__(self): 
        return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k%len(self.items)]
        return item['tid']

def rc_collate_fn(items):
    xx = nn.utils.rnn.pad_sequence(items, batch_first=True)
    return xx

def ee_collate_fn(items):
    xx = nn.utils.rnn.pad_sequence(items, batch_first=True)
    return xx

# Model
class RCModel(nn.Module):
    def __init__(self, plm_name, hidden_size, output_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained(plm_name)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        z = self.bert(x).last_hidden_state
        out = self.fc(z[:,0,:])
        out = torch.sigmoid(out)
        return out

class EEModel(nn.Module):
    def __init__(self, plm_name, hidden_size):
        super().__init__()
        self.bert = BertModel.from_pretrained(plm_name)
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, x):
        z = self.bert(x).last_hidden_state
        out = self.fc(z)
        out = torch.sigmoid(out)
        return out

# Metric
class MetricF1:
    def __init__(self):
        self.correct = self.output = self.golden = 0
    def append(self, out, ans):
        out, ans = set(out), set(ans)
        mid = out & ans
        self.correct += len(mid)
        self.output += len(out)
        self.golden += len(ans)

    def compute(self, show=True):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        if show: print(pstr)
        return f1

    # 为了绘制PR curve打印到文件上
    def compute_and_record(self, fout):
        correct, output, golden = self.correct, self.output, self.golden
        prec = correct / max(output, 1);  reca = correct / max(golden, 1);
        f1 = 2 * prec * reca / max(1e-9, prec + reca)
        pstr = 'Prec: %.4f %d/%d, Reca: %.4f %d/%d, F1: %.4f' % (prec, correct, output, reca, correct, golden, f1)
        fout.write(pstr+'\n')
        return (prec, reca, f1)

def tt(t):
    return t['p']+' | ' + t['s'] + ' | ' + t['o']

def ot(t):
    return t['type']+' | ' + t['head']['name'] + ' | ' + t['tail']['name']

sys.path.append('../')
import pt_utils

def ComputeOne(item, triples, f1, fout, label=None):
    #print(item)
    spos = [x for x in item['spo_list'] if x['p'] == item['label']]
    triples = set(tt(x) for x in triples)
    spos = set(tt(x) for x in spos)
    print('-'*30, file=fout)
    print(item['text'], file=fout)
    for x in triples&spos: print('o', x, file=fout)
    for x in triples-spos: print('-', x, file=fout)
    for x in spos-triples: print('+', x, file=fout)
    f1.append(triples, spos)

def test_ee(): 
    outs = [] 
    with torch.no_grad():
        for x, y in dl_dev:
            out = ee(x.cuda()).detach().cpu()
            for z in out: outs.append(z.numpy())
    f1 = MetricF1()
    fout = open(wdir('ret.txt'), 'w', encoding='utf-8')
    for item, rr in zip(dss['test'].items, outs):
        triples = decode_triples(item, rr, ee_threshold)
        ComputeOne(item, triples, f1, fout)
    f1.compute()
    fout.close()

def decode_triples(item, rr, ee_threshold):
    otokens = item['otokens']
    subs = GetTopSpans(otokens, rr[item['plen']:,:2])
    objs = GetTopSpans(otokens, rr[item['plen']:,2:])
    vv1 = [x for x,y in subs if y >= 0.1]
    vv2 = [x for x,y in objs if y >= 0.1]
    subv = {x:y for x,y in subs}
    objv = {x:y for x,y in objs}
    triples = []
    for sv1, sv2 in [(sv1, sv2) for sv1 in vv1 for sv2 in vv2]:
        score = min(subv[sv1], objv[sv2])
        if score < ee_threshold: continue
        triples.append( {'p':item['label'], 's': sv1, 'o':sv2} )
    return triples

def model_extract_pipeline(weight_file, labels):
    rel2id = {item: index for index, item in enumerate(labels)}
    id2rel = {v:k for k,v in rel2id.items()}

    rc_file = f'{weight_file}/rc.pt'
    ee_file = f'{weight_file}/ee.pt'

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    config = BertConfig.from_pretrained(args.model_path)

    rc = RCModel(plm_name = args.model_path, hidden_size = config.hidden_size, output_dim = len(labels)).cuda()
    #rc = nn.DataParallel(rc)
    rc.load_state_dict(torch.load(rc_file))
    ee = EEModel(plm_name = args.model_path, hidden_size = config.hidden_size).cuda()
    #ee = nn.DataParallel(ee)
    ee.load_state_dict(torch.load(ee_file))
    outs = []
    with open(os.path.join(args.test_file), 'r', encoding='utf-8') as f:
        tdata = json.load(f)

    ds_rc = Dataset(tdata, tokenizer, args.maxlen)
    dl_rc = torch.utils.data.DataLoader(ds_rc, batch_size=16, shuffle=False, collate_fn=rc_collate_fn)

    with torch.no_grad():
        for x in dl_rc:
            out = rc(x.cuda()).detach().cpu()
            for z in out: outs.append(z.numpy())
    f1 = MetricF1()
    for item, out in tqdm(zip(tdata, outs), desc='Handle RC process'):
        #print(item)
        rc_pred = []
        for i, v in enumerate(out):
            if v > rc_threshold: rc_pred.append(id2rel[i])
        item['rc_pred'] = rc_pred
        f1.append(rc_pred, list(set(x['type'] for x in item['relations'])))

    ds_ee = DatasetEE(tdata, tokenizer, args.maxlen)
    dl_ee = torch.utils.data.DataLoader(ds_ee, batch_size=16, shuffle=False, collate_fn=ee_collate_fn)
    outs = [] 
    with torch.no_grad():
        for x in tqdm(dl_ee, desc='Handle EE process'):
            out = ee(x.cuda()).detach().cpu()
            for z in out: outs.append(z.numpy())
    for item, rr in zip(ds_ee.items, outs):  
        triples = decode_triples(item, rr, ee_threshold)
        tdata[item['id']].setdefault('preds', []).extend(triples)

    return tdata

## Main Function
with open(os.path.join(args.train_dp, 'labels.json'), 'r', encoding='utf-8') as f:
    train_labels = json.load(f)

with open(os.path.join(args.test_dp, 'labels.json'), 'r', encoding='utf-8') as f:
    test_labels = json.load(f)

train_name = args.train_dp.split('/')[-1]
test_name = args.test_dp.split('/')[-1]

id2label = {i: label for i, label in enumerate(train_labels)}
defined_labels = list(set(train_labels) & set(test_labels))
print(defined_labels)

tdata = model_extract_pipeline(args.test_weight_file, train_labels)

fout = open(f'{args.result_path}/cross_{train_name}_{test_name}.txt', 'w', encoding='utf-8')
f1 = MetricF1()
for item in tdata:
    #print(item)
    triples, spos = item.get('preds', []), item['relations']
    triples = set(tt(x) for x in triples if x['p'] in defined_labels)
    spos = set(ot(x) for x in spos if x['type'] in defined_labels)
    print('-'*30, file=fout)
    for x in triples&spos: print('o', x, file=fout)
    for x in triples-spos: print('-', x, file=fout)
    for x in spos-triples: print('+', x, file=fout)
    f1.append(triples, spos)
print('='*30, file=fout)
print('Test result:', file=fout)
f1.compute_and_record(fout)

tdata = model_extract_pipeline(args.refer_weight_file, test_labels)
f1 = MetricF1()
for item in tdata:
    #print(item)
    triples, spos = item.get('preds', []), item['relations']
    triples = set(tt(x) for x in triples if x['p'] in defined_labels)
    spos = set(ot(x) for x in spos if x['type'] in defined_labels)
    f1.append(triples, spos)
print('='*30, file=fout)
print('Reference result:', file=fout)
f1.compute_and_record(fout)
fout.close()
