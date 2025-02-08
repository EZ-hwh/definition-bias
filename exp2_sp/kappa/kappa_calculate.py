import glob
import json, csv, os, yaml, re
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

# 假设我们有一个简单的数据框架，其中包含五个评估者的标注结果
# 1 表示识别了实体，0 表示没有识别
# 这只是一个示例，您需要用您自己的数据替换它

data = {
    'Dataset1': [1, 0, 1, 1, 0],
    'Dataset2': [1, 1, 1, 0, 0],
    'Dataset3': [1, 0, 1, 1, 1],
    'Dataset4': [0, 1, 1, 0, 0],
    'Dataset5': [1, 0, 1, 1, 0]
}

# 转换数据为适合 Fleiss Kappa 计算的格式
df = pd.DataFrame(data)
# 计算每行的总和，即每个实体被识别的次数
category_counts = df.apply(pd.Series.value_counts, axis=1).fillna(0)
# 调整列顺序以匹配 Fleiss Kappa 函数的期望格式（先0后1）
category_counts = category_counts[[0, 1]]

# 计算 Fleiss' Kappa
kappa = fleiss_kappa(category_counts)

kappa

yaml_path = '../test.yaml'
with open(yaml_path) as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)

def NER_parser(sentence):
    sentence = sentence.strip()
    item = sentence.split(';')
    return item

def RE_parser(sentence):
    sentence = sentence.strip()
    p1 = re.compile(r'[(](.*?)[)]', re.S)
    item = re.findall(p1, sentence)
    return item

def cal_fleiss_kappa(dataset):
    if dataset[0]['fake_dataset'] in dataset_config['NER']:
        func = NER_parser
    else:
        func = RE_parser

    dataset_set = set([data['fake_dataset'] for data in dataset])
    dataset_num = len(dataset_set)

    if dataset_num == 1:
        return None
    
    matrix = {data_name: [] for data_name in dataset_set}

    for index in range(len(dataset)//dataset_num):
        sub_dataset = dataset[index * dataset_num: (index + 1) * dataset_num]
        label_result_set = set()
        for data in sub_dataset:
            parser_result = func(data['pred'])
            data['parser_result'] = parser_result
            for item in parser_result:
                label_result_set.add(item)
        for data in sub_dataset:
            dataname = data['fake_dataset']
            for item in label_result_set:
                if item in data['parser_result']:
                    matrix[dataname].append(1)
                else:
                    matrix[dataname].append(0)

    # 转换数据为适合 Fleiss Kappa 计算的格式
    df = pd.DataFrame(matrix)
    category_counts = df.apply(lambda x: pd.Series.value_counts(x, dropna=False).reindex([0, 1], fill_value=0), axis=1)
    #print(category_counts)
    # 调整列顺序以匹配 Fleiss Kappa 函数的期望格式（先0后1）
    category_counts = category_counts[[0, 1]]

    # 计算 Fleiss' Kappa
    kappa = fleiss_kappa(category_counts)
    return kappa

with open('type_kappa.json', 'w') as fout:
    output = {}
    for file in sorted(glob.glob('label_catogory/*.json')):
        label_name = file.split('/')[-1].replace('.json', '')
        print(label_name)
        with open(file, 'r') as f:
            dataset = json.load(f)
        try:
            kappa = cal_fleiss_kappa(dataset)
            output[label_name] = kappa
        except:
            output[label_name] = 1
    json.dump(output, fout, ensure_ascii=False, indent=4)