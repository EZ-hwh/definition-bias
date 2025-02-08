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

parser = argparse.ArgumentParser()
parser.add_argument("--dname", help='dataset for training and testing')
parser.add_argument("--model_path", help='the path to the pretrained model')
parser.add_argument("--data_path", help='the path to the data directory')
parser.add_argument("--save_path", help='the path to the save directory')
parser.add_argument("--do_train", help='training model for both RC and EE model', type=bool, default=False)
parser.add_argument("--do_test", help='test model as a pipeline', type=bool, default=False)
parser.add_argument("--filter", help='whether use triple filter module',type=bool, default=False)
parser.add_argument("--maxlen", help='the max length of the input', type=int)

args = parser.parse_args()
dname = args.dname
dsplits = 'train test'.split()

def wdir(x): return os.path.join(args.save_path, x)
def ddir(x): return os.path.join(args.data_path, x)
fns = {x:ddir(f'{x}.json') for x in dsplits}

# rc_threshold = config[dname]['thre_rc']
# ee_threshold = config[dname]['thre_ee']
rc_threshold = 0.5
ee_threshold = 0.5

from transformers import BertTokenizer, BertModel, set_seed, BertConfig
set_seed(42)

tokenizer = BertTokenizer.from_pretrained(args.model_path)
config = BertConfig.from_pretrained(args.model_path)

def loadjson(file_name):
    with open(file_name) as f:
        return json.load(f)
#loadjson = lambda x: with open(x) as f: return json.load(f)

with open(ddir('labels.json')) as fin:
    rel_list = json.load(fin)
    print(rel_list)
    rel2id = {item: index for index, item in enumerate(rel_list)}

id2rel = {v:k for k,v in rel2id.items()}

rels = None

from utils import TN, restore_token_list, GetTopSpans, FindValuePos

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.y = 0
        #global rel
        global rel2id
        self.items = []
        self.maxlen = maxlen
        for z in data:
            item = {}
            item['tid'] = torch.tensor(tokenizer.encode(z['sentence'])[:maxlen])
            item['yrc'] = list(set(rel2id[x['type']] for x in z['relations']))
            self.items.append(item)
            self.y += len(item['yrc'])
    def __len__(self): return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k]
        return item['tid'], item['yrc']

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
                slen = item['tid'].size(0)
                ss = set(TN(x['s']) for x in spo_list if x['p'] == label)
                oo = set(TN(x['o']) for x in spo_list if x['p'] == label)
                yy = torch.zeros((slen, 4)).float()
                for s in ss:
                    for u, v in FindValuePos(otokens, s):
                        if v-1+plen < maxlen:
                            yy[u+plen,0] = yy[v-1+plen,1] = 1
                for o in oo:
                    for u, v in FindValuePos(otokens, o):
                        if v-1+plen < maxlen:
                            yy[u+plen,2] = yy[v-1+plen,3] = 1
                item['yy'] = yy
                self.items.append(item)
    def __len__(self): 
        return len(self.items)
    def __getitem__(self, k): 
        item = self.items[k%len(self.items)]
        return item['tid'], item['yy']

def rc_collate_fn(items):
    xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)
    yy = torch.zeros((len(items), len(rel_list)))
    for i, (x, ys) in enumerate(items):
        for y in ys: yy[i,y] = 1
    return xx, yy

def ee_collate_fn(items):
    xx = nn.utils.rnn.pad_sequence([x for x,y in items], batch_first=True)
    yy = nn.utils.rnn.pad_sequence([y for x,y in items], batch_first=True)
    return xx, yy.float()


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


def decode_triples(item, rr, ee_threshold, gpout=None):
    otokens = item['otokens']
    subs = GetTopSpans(otokens, rr[item['plen']:,:2])
    objs = GetTopSpans(otokens, rr[item['plen']:,2:])
    vv1 = [x for x,y in subs if y >= 0.1]
    vv2 = [x for x,y in objs if y >= 0.1]
    subv = {x:y for x,y in subs}
    objv = {x:y for x,y in objs}
    triples = []
    for sv1, sv2 in [(sv1, sv2) for sv1 in vv1 for sv2 in vv2]:
        # if gpout is not None:
        #     loc1, loc2 = FindValuePos(otokens, sv1), FindValuePos(otokens, sv2)
        #     vals = []
        #     for u1, v1 in loc1:
        #         for u2, v2 in loc2:
        #             vals.append([])
        #             for i in range(1+u1, 1+v1):
        #                 for j in range(1+u2, 1+v2):
        #                     vals[-1].append(gpout[0,i,j])
        #     ind = item['id']
        #     tdata[ind].setdefault('gp_detail', []).append( (sv1, sv2, vals) )
        #     vals = [np.array(x).mean() for x in vals]
        #     tdata[ind].setdefault('gp', []).append( (sv1, sv2, vals) )
        #     if len(vals) == 0: continue
        #     if max(vals) < 0:
        #         continue
        score = min(subv[sv1], objv[sv2])
        if score < ee_threshold: continue
        triples.append( {'p':item['label'], 's': sv1, 'o':sv2} )
    return triples

# Train RC
if args.do_train:
    epochs = 20
    print(len(rel_list))
    dss = {x:Dataset(loadjson(fn), tokenizer, args.maxlen) for x, fn in fns.items()}
    dl_train = torch.utils.data.DataLoader(dss['train'], batch_size=16 * torch.cuda.device_count(), shuffle=True, collate_fn=rc_collate_fn)
    dl_dev = torch.utils.data.DataLoader(dss['test'], batch_size=16 * torch.cuda.device_count(), shuffle=False, collate_fn=rc_collate_fn)
    total_steps = len(dl_train) * epochs
    rc = RCModel(plm_name = args.model_path, hidden_size = config.hidden_size, output_dim = len(rel_list)).cuda()
    pt_utils.lock_transformer_layers(rc.bert, 6)
    # FIXME: 多卡训练
    rc = nn.DataParallel(rc)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    rcmfile = wdir(f'rc.pt')
    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(rc, 2e-5, total_steps)

    loss_fct = nn.BCELoss()
    
    FN_RATIO = 0
    
    #loss_fct = lambda y_pred, y_true: - 10*(y_true*torch.log(y_pred+1e-9)).mean() - torch.log(((1-y_true)*(1-y_pred)).mean()+1e-9)
    #print(dss['train'].y)
    PI_RC = dss['train'].y / (len(dss['train']) * len(rel2id))
    #print(PI_RC)
    pu_loss_fct = PU_mid_loss(mid=PI_RC*(1+FN_RATIO),pi=1).cuda()


    def train_func(model, ditem):
        x, y = ditem
        y = y.cuda()
        out = model(x.cuda())
        loss = loss_fct(out, y) + 0.1 * out.mean()
        #loss = pu_loss_fct(y,out) #+ 0.1 * out.mean()
        #print(y.device, out.device)
        oc = (out > 0.5).float()
        prec = (oc + y > 1.5).sum() / max(oc.sum().item(), 1)
        reca = (oc + y > 1.5).sum() / max(y.sum().item(), 1)
        f1 = 2 * prec * reca / (prec + reca)
        r = {'loss': loss, 'prec': prec, 'reca': reca, 'f1':f1}
        return r

    def test_func(): 
        outs = [];  ys = []
        for x, y in dl_dev:
            out = (rc(x.cuda()) > 0.5).long().detach().cpu()
            outs.append(out)
            ys.append(y)
        outs = torch.cat(outs, 0)
        ys = torch.cat(ys, 0)
        accu = (outs == ys).float().mean()
        prec = (outs + ys == 2).float().sum() / outs.sum()
        reca = (outs + ys == 2).float().sum() / ys.sum()
        f1 = 2 * prec * reca / (prec + reca)
        print(f'Accu: {accu:.4f},  Prec: {(outs + ys == 2).float().sum()} / {outs.sum()}:{prec:.4f},  Reca: {(outs + ys == 2).float().sum()} / {ys.sum()}:{reca:.4f},  F1: {f1:.4f}')

    pt_utils.train_model(rc, optimizer, dl_train, epochs, train_func, test_func, 
                   scheduler=scheduler, save_file=rcmfile)

# Train ee
if args.do_train:
    epochs = 20
    dss = {x:DatasetEE(loadjson(fn), tokenizer, args.maxlen) for x, fn in fns.items()}
    dl_train = torch.utils.data.DataLoader(dss['train'], batch_size=16 * torch.cuda.device_count(), shuffle=True, collate_fn=ee_collate_fn)
    dl_dev = torch.utils.data.DataLoader(dss['test'], batch_size=16 * torch.cuda.device_count(), shuffle=False, collate_fn=ee_collate_fn)
    total_steps = len(dl_train) * epochs

    ee = EEModel(plm_name = args.model_path, hidden_size = config.hidden_size).cuda()
    pt_utils.lock_transformer_layers(ee.bert, 6)
    # FIXME: 多卡训练
    ee = nn.DataParallel(ee)

    eemfile = wdir(f'ee.pt')
    #ee.load_state_dict(torch.load(eemfile))
    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(ee, 2e-5, total_steps)

    loss_fct = lambda y_pred, y_true: - (y_true*torch.log(y_pred+1e-9) + (1-y_true)*torch.log(1-y_pred+1e-9)).mean()
    #loss_fct = lambda y_pred, y_true: - 5*(y_true*torch.log(y_pred+1e-9)).mean() - torch.log(((1-y_true)*(1-y_pred)).mean()+1e-9)

    def train_func(model, ditem):
        x, y = ditem
        y = y.cuda()
        out = model(x.cuda())
        loss = loss_fct(out, y)# + 0.1 * out.mean()
        oc = (out > 0.5).float()
        prec = (oc + y > 1.5).sum() / max(oc.sum().item(), 1)
        reca = (oc + y > 1.5).sum() / max(y.sum().item(), 1)
        f1 = 2 * prec * reca / (prec + reca)
        r = {'loss': loss, 'prec': prec, 'reca': reca, 'f1':f1}
        return r

    pt_utils.train_model(ee, optimizer, dl_train, epochs, train_func, test_ee, 
                   scheduler=scheduler, save_file=eemfile)

if args.do_test:
    tdata = loadjson(fns['test'])
    ds_rc = Dataset(tdata, tokenizer, args.maxlen)
    dl_rc = torch.utils.data.DataLoader(ds_rc, batch_size=16, shuffle=False, collate_fn=rc_collate_fn)
    rc = RCModel(plm_name = args.model_path, hidden_size = config.hidden_size, output_dim = len(rel_list)).cuda()
    #rc = nn.DataParallel(rc)
    rc.load_state_dict(torch.load(wdir(f'rc.pt')))
    ee = EEModel(plm_name = args.model_path, hidden_size = config.hidden_size).cuda()
    #ee = nn.DataParallel(ee)
    ee.load_state_dict(torch.load(wdir(f'ee.pt')))
    outs = [] 
    with torch.no_grad():
        for x, y in dl_rc:
            out = rc(x.cuda()).detach().cpu()
            for z in out: outs.append(z.numpy())
    f1 = MetricF1()
    for item, out in tqdm(zip(tdata, outs)):
        #print(item)
        rc_pred = []
        for i, v in enumerate(out):
            if v > rc_threshold: rc_pred.append(id2rel[i])
        item['rc_pred'] = rc_pred
        f1.append(rc_pred, list(set(x['type'] for x in item['relations'])))
    print('\n')

    ds_ee = DatasetEE(tdata, tokenizer, args.maxlen)
    dl_ee = torch.utils.data.DataLoader(ds_ee, batch_size=30, shuffle=False, collate_fn=ee_collate_fn)
    outs = [] 
    with torch.no_grad():
        for x, y in dl_ee:
            out = ee(x.cuda()).detach().cpu()
            for z in out: outs.append(z.numpy())
    for item, rr in zip(ds_ee.items, outs):  
        triples = decode_triples(item, rr, ee_threshold)
        tdata[item['id']].setdefault('preds', []).extend(triples)
    
    f1 = MetricF1()
    fout = open(wdir(f'{dname}_ret.txt'), 'w', encoding='utf-8')
    for item in tdata:
        #print(item)
        triples, spos = item.get('preds', []), item['relations']
        triples = set(tt(x) for x in triples)
        spos = set(ot(x) for x in spos)
        print('-'*30, file=fout)
        #print(item['spo_list'], file=fout)
        for x in triples&spos: print('o', x, file=fout)
        for x in triples-spos: print('-', x, file=fout)
        for x in spos-triples: print('+', x, file=fout)
        f1.append(triples, spos)
    #print(f'\ntextlen={textlen}')
    f1.compute_and_record(fout)
    fout.close()

