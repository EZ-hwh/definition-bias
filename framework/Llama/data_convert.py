import os, json, yaml
from data.data_utils import WholeLoader
import math


#Params
file_name = 'weighted.jsonl'

origin_loader = WholeLoader('../../data/IE_INSTRUCTIONS', 'train.yaml')

with open('train.yaml') as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)

origin_dataset = origin_loader.get_dataset(data_split=['train'], max_samples_per_dataset=10000)

template_wo_example = 'Instruction:{0}\n\nInput:{1}\nOutput:'

#{"items":[{"role":"user","content":"What is C-RLFT?","weight":0.0},{"role":"assistant","content":"I don't know.","weight":0.1}],"condition":"GPT3","system":""}
with open('measure/type_kappa.json', 'r') as f:
    type_kappa = json.load(f)
    for key in type_kappa.keys():
        if type_kappa[key]:
            if math.isnan(type_kappa[key]):
                type_kappa[key] = 0.5
            type_kappa[key] = max(type_kappa[key], 0.1)
        else:
            type_kappa[key] = 0.1
    #type_kappa = math.isnan(x)
print(type_kappa)

with open('measure/gpt_kappa.json', 'r') as f:
    gpt_kappa = json.load(f)
    for key in gpt_kappa.keys():
        gpt_kappa[key] += 1

    NER_avg = sum([value for key, value in gpt_kappa.items() if key in dataset_config['NER']]) / 8
    RE_avg = sum([value for key, value in gpt_kappa.items() if key in dataset_config['RE']]) / 5
    # Adjust RE weight
    for key in gpt_kappa.keys():
        if key in dataset_config['RE']:
            gpt_kappa[key] = gpt_kappa[key] * NER_avg / RE_avg

print(gpt_kappa)

with open(os.path.join('dataset', file_name), 'w') as f:
    for data in origin_dataset:
        f.write(json.dumps({
            "items": [
                {
                    "role": "", 
                    "content": template_wo_example.format(data['instruction'], data['input']),
                    #"content": template_fs_example.format(data['instruction'], data['few_shot_case'], data['input']),
                    "weight": 0.0 
                },
                {   "role": "", 
                    "content": data['output'],
                    # "weight": 1.0
                    "weight": 2 * max(0.1, gpt_kappa.get(data['dataset_name'], 0.4) * sum([type_kappa.get(item, 0.1) for item in data['label_list']]) / (len(data['label_list']) + 1e-6))
                }
            ],
            "condition": "",
            "system": ""
        }) + '\n')
        