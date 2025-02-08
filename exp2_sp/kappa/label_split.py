import csv, json, yaml, random
import os
from collections import defaultdict

yaml_path = '../test.yaml'
data_path = '../../../data/IE_INSTRUCTIONS'
label_path = 'label_catogory'

dataset_label = {}
schema_dict = defaultdict(list)

with open(yaml_path) as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    for task_type in dataset_config:
        for dataset_name in dataset_config[task_type]:
            with open(os.path.join(data_path, task_type, dataset_name, 'labels.json')) as f1:
                dataset_label[dataset_name] = json.load(f1)

print(dataset_label)

with open("source_prompt_v1.json") as f:
    dataset = json.load(f)
    for data in dataset:
        if data['label'] in dataset_label[data['fake_dataset']]:
            schema_dict[data['label']].append(data)

for key in schema_dict.keys():
    with open(os.path.join(label_path, f'{key}.json'), 'w') as f:
        json.dump(schema_dict[key], f, ensure_ascii=False, indent=4)
