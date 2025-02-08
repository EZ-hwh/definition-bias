import json, yaml
import re
from collections import defaultdict

yaml_file = 'train.yaml'
file_name = 'result/flant5_large_true_source.json'
output_name = 'result/flant5_large_true_source.txt'

with open(yaml_file) as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)

#print(dataset_config)

def NER_parser(sentence):
    sentence = sentence.strip()
    item = sentence.split(';')
    return item

def RE_parser(sentence):
    sentence = sentence.strip()
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    item = re.findall(p1, sentence)
    return item

with open(file_name, 'r') as f:
    datasets = json.load(f)

tp, tn, fp = defaultdict(int), defaultdict(int), defaultdict(int)

output_stream = open(output_name, 'w')

for data in datasets:
    #print(data)
    data_name = data['dataset']
    if data_name in dataset_config['NER']:
        gt = set(NER_parser(data['output']))
        pred = set(NER_parser(data['pred']))
        tp[data_name] += len(gt & pred)
        tn[data_name] += len(pred - gt)
        fp[data_name] += len(gt - pred)
    else:
        gt = set(RE_parser(data['output']))
        pred = set(RE_parser(data['pred']))
        tp[data_name] += len(gt & pred)
        tn[data_name] += len(pred - gt)
        fp[data_name] += len(gt - pred)

print('NER', file=output_stream)
print('-'*50, file=output_stream)
for data_name in dataset_config['NER']:
    prec = tp[data_name] / (tp[data_name] + tn[data_name] + 1e-12)
    reca = tp[data_name] / (tp[data_name] + fp[data_name] + 1e-12)
    f1 = 2 * prec * reca / (prec + reca + 1e-12)
    print("Dataset name: %s\tPrec: %.4f\tReca: %.4f\tF1: %.4f"%(data_name, prec, reca, f1), file=output_stream)
print(file=output_stream)
print('RE', file=output_stream)
print('-'*50, file=output_stream)
for data_name in dataset_config['RE']:
    prec = tp[data_name] / (tp[data_name] + tn[data_name] + 1e-12)
    reca = tp[data_name] / (tp[data_name] + fp[data_name] + 1e-12)
    f1 = 2 * prec * reca / (prec + reca + 1e-12)
    print("Dataset name: %s\tPrec: %.4f\tReca: %.4f\tF1: %.4f"%(data_name, prec, reca, f1), file=output_stream)