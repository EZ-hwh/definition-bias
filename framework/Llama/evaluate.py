import json, yaml
import re
from collections import defaultdict

yaml_file = 'train.yaml'
dataset_name = 'TweetNER7'
model_name = 'main_1214'
file_name = f'result/{model_name}/{dataset_name}.json'
output_name = f'result/{model_name}/{dataset_name}.txt'

with open(yaml_file) as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)

#print(dataset_config)
def prettify_text(s):
    s = ' '.join(s.split())     # 多个空白字符变为单个空格
    s = re.sub(r"\s*(,|:|\(|\)|\.|_|;|'|-)\s*", r'\1', s)   #去除特殊符号旁的空白字符
    s = s.lower()
    s = s.replace('{','').replace('}','')
    s = re.sub(',+', ',', s)
    s = re.sub('\.+', '.', s)
    s = re.sub(';+', ';', s)
    s = s.replace('’', "'")
    s = s.replace('location', 'located')
    return s

def NER_parser(sentence):
    #sentence = sentence.replace('\n','').replace('the ','').strip()
    sentence = prettify_text(sentence)
    item = sentence.split(';')
    item = [i.strip() for i in item if 'none' not in i.lower() and len(i.strip()) and ":" in i]
    item = [prettify_text(i) for i in item]
    return item

def RE_parser(sentence):
    sentence = sentence.strip().replace("'", "")
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    item = re.findall(p1, sentence)
    item = [i.strip() for i in item]
    return item

with open(file_name, 'r') as f:
    datasets = json.load(f)
# datasets = []
# with open(file_name, 'r') as f:
#     for line in f.readlines():
#         datasets.append(json.loads(line))

tp, tn, fp = defaultdict(int), defaultdict(int), defaultdict(int)
output_stream = open(output_name, 'w')

# data_name = 't'

# for data in datasets:
#     gt = set(NER_parser(data["Instance"]['ground_truth']))
#     pred = set(NER_parser(data['Prediction']))
#     for item in gt & pred: print('o', item, file=output_stream)
#     for item in pred - gt: print('-', item, file=output_stream)
#     for item in gt - pred: print('+', item, file=output_stream)
#     tp[data_name] += len(gt & pred)
#     tn[data_name] += len(pred - gt)
#     fp[data_name] += len(gt - pred)
# for data_name in ['t']:
#     prec = tp[data_name] / (tp[data_name] + tn[data_name] + 1e-12)
#     reca = tp[data_name] / (tp[data_name] + fp[data_name] + 1e-12)
#     f1 = 2 * prec * reca / (prec + reca + 1e-12)
#     print("Dataset name: %s\tPrec: %.4f\tReca: %.4f\tF1: %.4f"%(data_name, prec, reca, f1))

for data in datasets:
    #print(data)
    print('-'*30, file=output_stream)
    data_name = data['dataset_name']
    if data_name in dataset_config['NER']:
        gt = set(NER_parser(data['output']))
        pred = set(NER_parser(data['pred']))
        for item in gt & pred: print('o', item, file=output_stream)
        for item in pred - gt: print('-', item, file=output_stream)
        for item in gt - pred: print('+', item, file=output_stream)
        tp[data_name] += len(gt & pred)
        tn[data_name] += len(pred - gt)
        fp[data_name] += len(gt - pred)
    else:
        gt = set(RE_parser(data['output']))
        pred = set(RE_parser(data['pred']))
        for item in gt & pred: print('o', item, file=output_stream)
        for item in pred - gt: print('-', item, file=output_stream)
        for item in gt - pred: print('+', item, file=output_stream)
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
