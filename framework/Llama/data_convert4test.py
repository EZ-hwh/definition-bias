import os, json, yaml
from data.data_utils import WholeLoader

#Params
file_name = 'test_fs.jsonl'

origin_loader = WholeLoader('../../data/IE_INSTRUCTIONS', 'train.yaml')
origin_dataset = origin_loader.get_dataset(data_split=['test'], few_shot=4)

template_wo_example = 'Instruction:{0}\n\nInput:{1}\nOutput:'
template_fs_example = 'Instruction:{0}\n\n{1}\n\nInput:{2}\nOutput:'

with open(os.path.join('dataset', file_name), 'w') as f:
    for data in origin_dataset:
        f.write(json.dumps({
            "items": [
                {
                    "role": "user", 
                    #"content": template_wo_example.format(data['instruction'], data['input']),
                    "content": template_fs_example.format(data['instruction'], data['few_shot_case'], data['input']),
                    "weight": 0.0 
                },
                {   "role": "assistant", 
                    "content": data['output'],
                    "weight": 1.0 
                }
            ],
            "condition": data["dataset_name"],
            "system": ""
        }) + '\n')
        #{"items":[{"role":"user","content":"What is C-RLFT?","weight":0.0},{"role":"assistant","content":"I don't know.","weight":0.1}],"condition":"GPT3","system":""}