import json, yaml
import re
import pandas as pd
from collections import defaultdict
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

file_name = 'gpt4_fs_prompt.jsonl'
yaml_file = '../test.yaml'

with open(yaml_file) as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)

def NER_parser(sentence):
    sentence = sentence.strip().replace('\n','')
    item = sentence.split(';')
    item = [i.strip() for i in item if 'none' not in i.lower() and len(i.strip()) and ":" in i]
    item = [i for i in item if i.split(':')[1].strip() != '']
    return item

def RE_parser(sentence):
    sentence = sentence.strip().replace("'", "")
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    item = re.findall(p1, sentence)
    item = [i.strip() for i in item if 'none' not in i.lower()]
    return item

with open(file_name, 'r') as f:
    datasets = []
    for line in f.readlines():
        datasets.append(json.loads(line))

dataset_kappa_record = {}

def cal_fleiss_kappa(record):
    df = pd.DataFrame(record)
    category_counts = df.apply(lambda x: pd.Series.value_counts(x, dropna=False).reindex([0, 1], fill_value=0), axis=1)
    #print(category_counts)
    # 调整列顺序以匹配 Fleiss Kappa 函数的期望格式（先0后1）
    category_counts = category_counts[[0, 1]]

    # 计算 Fleiss' Kappa
    kappa = fleiss_kappa(category_counts)
    return kappa

for data in datasets:
    data_name = data['condition']
    if data_name not in dataset_kappa_record.keys():
        dataset_kappa_record[data_name] = {
            'dataset': [],
            'gpt4': []
        }
    if data_name in dataset_config['NER']:
        gt = set(NER_parser(data['items'][1]['content']))
        pred = set(NER_parser(data['output']))
        for entity in gt | pred:
            if entity in pred:
                dataset_kappa_record[data_name]['gpt4'].append(1)
            else:
                dataset_kappa_record[data_name]['gpt4'].append(0)
            if entity in gt:
                dataset_kappa_record[data_name]['dataset'].append(1)
            else:
                dataset_kappa_record[data_name]['dataset'].append(0)
    if data_name in dataset_config['RE']:
        gt = set(RE_parser(data['items'][1]['content']))
        pred = set(RE_parser(data['output']))
        for entity in gt | pred:
            if entity in pred:
                dataset_kappa_record[data_name]['gpt4'].append(1)
            else:
                dataset_kappa_record[data_name]['gpt4'].append(0)
            if entity in gt:
                dataset_kappa_record[data_name]['dataset'].append(1)
            else:
                dataset_kappa_record[data_name]['dataset'].append(0)

kappa_result = {}
with open('gpt_kappa.json', 'w') as f:
    for key in dataset_kappa_record.keys():
        kappa_result[key] = cal_fleiss_kappa(dataset_kappa_record[key])
        #print(dataset_kappa_record[key])
        #print(key, cal_fleiss_kappa(dataset_kappa_record[key]))
    json.dump(kappa_result,f,ensure_ascii=False, indent=4)