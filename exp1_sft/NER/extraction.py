import os, sys, json
from tqdm import tqdm
import argparse
import torch.nn as nn
import torch
from transformers import BertTokenizerFast, RobertaTokenizerFast, set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

# Self package
sys.path.append('../')

from model import GlobalPointerModel, global_pointer_crossentropy, global_pointer_f1_score
from dataset import NERDataset, collate_fn_cuda
import pt_utils
from collections import defaultdict

# Global params
LR = 2e-5
batch_size = 12
epochs = 20
set_seed(42)
#defined_labels = ["location", "else", "organization", "person"]


# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", help='training extraction model', type=bool, default=False)
parser.add_argument("--do_test", help='predict the knowledge tuple in texts.', type=bool, default=False)
parser.add_argument("--train_dp", help='path to the data file.', default='/mnt/huangwenhao/data122/datasets/Information Extraction/academic_dataset/IE_INSTRUCTIONS/NER/ACE 2004', type=str)
parser.add_argument("--test_dp", help='path to the data file.', default='/mnt/huangwenhao/data122/datasets/Information Extraction/academic_dataset/IE_INSTRUCTIONS/NER/ACE 2004', type=str)
parser.add_argument("--plm", help='backbone pretrained language model', default='bert-large-cased')
parser.add_argument("--weight_file", help='path to save the model weight.', default='gp_webnlg.pt', type=str)
parser.add_argument("--test_file", type=str, help='the file to be tested')
parser.add_argument("--result_path", help='path to save the model weight.', type=str)
args = parser.parse_args()

accelerator = Accelerator(mixed_precision='fp16', cpu=False)

if args.do_train:
    weight_file = f'weight/{args.weight_file}'
    tokenizer = BertTokenizerFast.from_pretrained(args.plm)
    defined_labels = []
    with open(os.path.join(args.train_dp, 'labels.json'), 'r', encoding='utf-8') as f:
        defined_labels = json.load(f)

    id2label = {i: label for i, label in enumerate(defined_labels)}

    train_ds = NERDataset(os.path.join(args.train_dp, 'train.json'), tokenizer, defined_labels)


    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_cuda)
    total_steps = len(train_dl) * epochs
    model = GlobalPointerModel(args.plm, len(defined_labels)).cuda()

    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, LR, total_steps)

    pt_utils.lock_transformer_layers(model.bert, 6)

    loss_fct = global_pointer_crossentropy
    def train_func(model, batch):
        input_ids, token_type_ids, labels = batch
        
        inputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids
        }
        logits = model(**inputs)
        loss = loss_fct(labels, logits)
        f1 = global_pointer_f1_score(labels, logits)
        return {'loss': loss, 'f1': f1}

    pt_utils.train_model(model, optimizer, train_dl, epochs, train_func, None,
                    scheduler=scheduler, save_file=weight_file, accelerator=accelerator)

def recognize(text, logits, offset_mapping, id2label, defined_labels, threshold = 0):
    import numpy as np
    '''
    根据GlobalPointer模型的输出进行抽取结果的识别，返回形式：
        [(entity, scores),...]
    '''
    #print(logits.size())
    scores = logits.cpu() # Size: [cls_num, seqlen, seqlen]
    entities = defaultdict(list)
    entity_list = []
    for index, start, end in zip(*np.where(scores > threshold)):
        entities[id2label[index]].append(
            (text[offset_mapping[start][0]: offset_mapping[end][1]], scores[index,start,end])
        )
        if id2label[index] in defined_labels:
            entity_list.append(f'{id2label[index]}|{text[offset_mapping[start][0]: offset_mapping[end][1]]}')
    return entities, entity_list


if args.do_test:
    def ner_pipeline(weight_file, labels, fout, test_case):
        id2label = {i: label for i, label in enumerate(labels)}

        weight_file = f'weight/{weight_file}'
        tokenizer = BertTokenizerFast.from_pretrained(args.plm)
        model = GlobalPointerModel(args.plm, len(labels)).cuda()
        model.load_state_dict(torch.load(weight_file))
        dataset = []
        with open(args.test_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        tp = 0
        fp = 0
        fn = 0

        for data in tqdm(dataset):
            output = tokenizer(data['sentence'], return_offsets_mapping=True, return_token_type_ids=True, max_length=512)
            input_ids = output['input_ids']
            token_type_ids = output['token_type_ids']
            labels = torch.zeros(len(defined_labels),len(input_ids), len(input_ids)).int()
            offset_mapping = output['offset_mapping']
            with torch.no_grad():
                inputs = {
                    'input_ids': torch.IntTensor(input_ids).unsqueeze(0).cuda(),
                    'token_type_ids': torch.IntTensor(token_type_ids).unsqueeze(0).cuda()
                }
                logits = model(**inputs)
                
                enitities, pred_entities = recognize(data['sentence'], logits[0], offset_mapping, id2label, defined_labels)
                gt_entities = []
                for entity in data['entities']:
                    if entity['type'] not in defined_labels:
                        continue
                    gt_entities.append(f"{entity['type']}|{entity['name']}")
                tp_set = set(gt_entities) & set(pred_entities)
                fp_set = set(pred_entities) - set(gt_entities)
                fn_set = set(gt_entities) - set(pred_entities)
                if test_case:
                    print('-'*30, file=fout)
                    print(data['sentence'], file=fout)
                    for x in tp_set: print('o', x, file=fout)
                    for x in fp_set: print('-', x, file=fout)
                    for x in fn_set: print('+', x, file=fout)
                tp += len(tp_set)
                fp += len(fp_set)
                fn += len(fn_set)
        print('='*30, file=fout)
        if test_case:
            print('Test result:', file=fout)
        else:
            print('Reference result:', file=fout)
        print(f'Precision: {tp/(tp+fp)}', file=fout)
        print(f'Recall: {tp/(tp+fn)}', file=fout)
        print(f'F1: {2*tp/(2*tp+fp+fn)}', file=fout)


    with open(os.path.join(args.train_dp, 'labels.json'), 'r', encoding='utf-8') as f:
        train_labels = json.load(f)
    with open(os.path.join(args.test_dp, 'labels.json'), 'r', encoding='utf-8') as f:
        test_labels = json.load(f)
    #print(train_labels)
    train_name = args.train_dp.split('/')[-1]
    test_name = args.test_dp.split('/')[-1]

    defined_labels = list(set(train_labels) & set(test_labels))
    print(defined_labels)
    if len(defined_labels) == 0:
        pass
    else:
        if not os.path.exists(args.result_path):
            os.mkdir(args.result_path)
        if not os.path.exists(f'{args.result_path}/{train_name}_{test_name}.txt'):
            fout = open(f'{args.result_path}/{train_name}_{test_name}.txt', 'w')

            ner_pipeline(f'gp_{train_name}.pt', train_labels, fout, True)
            ner_pipeline(f'gp_{test_name}.pt', test_labels, fout, False)
            
            fout.close()