import os, json, yaml
from data.data_utils import WholeLoader, NERLoader, RELoader
import math


#Params
file_name = 'train_1212.jsonl'
data_path = '../../data/IE_INSTRUCTIONS'
data_split = ['train']
#origin_loader = WholeLoader('../../data/IE_INSTRUCTIONS', 'train.yaml')

with open('train.yaml') as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)

#origin_dataset = origin_loader.get_dataset(data_split=['train'])

template_wo_example = 'Instruction:{0}\n\nInput:{1}\nOutput:'


for task in dataset_config.keys():
    for dataset in dataset_config[task]:
        if task == 'NER':
            dataloader = NERLoader(os.path.join(data_path, task, dataset), data_split)
        elif task == 'RE':
            dataloader = RELoader(os.path.join(data_path, task, dataset), data_split)
        inst_dataset = dataloader.get_dataset()

        with open(os.path.join('dataset', 'single', f'{dataset}.jsonl'), 'w') as f:
            for data in inst_dataset:
                f.write(json.dumps({
                    "items": [
                        {
                            "role": "user", 
                            "content": template_wo_example.format(data['instruction'], data['input']),
                            #"content": template_fs_example.format(data['instruction'], data['few_shot_case'], data['input']),
                            "weight": 0.0 
                        },
                        {   "role": "assistant", 
                            "content": data['output'],
                            "weight": 1
                        }
                    ],
                    "condition": data["dataset_name"],
                    "system": ""
                }) + '\n')
        