from vllm import LLM, SamplingParams
from data.data_utils import WholeLoader
from copy import deepcopy
import os

import torch
import csv, json, yaml, random
#from transformers import AutoModelForCausalLM, T5Tokenizer, LlamaTokenizer, AutoConfig, GenerationConfig
from tqdm import tqdm

yaml_path = 'test.yaml'
BASE_MODEL = "ckp/llama13b_source_nickname/ckp_epoch4"
data_path = '../../data/IE_INSTRUCTIONS'

dataset_label = {}
with open(yaml_path) as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    for task_type in dataset_config:
        for dataset_name in dataset_config[task_type]:
            with open(os.path.join(data_path, task_type, dataset_name, 'labels.json')) as f1:
                dataset_label[dataset_name] = json.load(f1)

llm = LLM(model=BASE_MODEL, dtype='float16', tensor_parallel_size=torch.cuda.device_count())


template_wo_example = 'Instruction:{0}\n\nInput:{1}\nOutput:'

def generate_prompt(instruction, text=None):
    return template_wo_example.format(instruction, text)

def evaluate(
    instruction,
    inputs,
    temperature=0.7,
    top_p=0.7,
    top_k=40,
    num_beams=1,
    length_penalty=1,
    repetition_penalty=1,
    **kwargs,
):
    sampling_params = SamplingParams(temperature=temperature, 
                   top_p=top_p,
                   top_k=top_k,
                   n=num_beams,
                   stop='<end>',
                   length_penalty=length_penalty,
                   max_tokens=128)

    text = generate_prompt(instruction, inputs)
    #print(prompt)
    output = llm.generate(text, sampling_params, use_tqdm=False)[0].outputs[0].text
    return output

origin_dataloader = WholeLoader(data_path, yaml_path)
origin_dataset = origin_dataloader.get_dataset(['val'], shuffle=False)

def make_false_negative_sample(instruction, origin_dataset, new_dataset):
    return instruction.replace(origin_dataset, new_dataset), new_dataset
    
new_dataset = []
for data in tqdm(origin_dataset):
    if data['dataset'] in dataset_config['NER']:
        for dataset in dataset_config['NER']:
            if data['label'] in dataset_label[dataset]:
                new_data = deepcopy(data)
                new_data['instruction'], new_data['fake_dataset'] = make_false_negative_sample(new_data['instruction'], new_data['dataset'], dataset)
                output = evaluate(new_data['instruction'], new_data['input'])
                new_data['pred'] = output
                new_dataset.append(new_data)
    else:
        for dataset in dataset_config['RE']:
            if data['label'] in dataset_label[dataset]:
                new_data = deepcopy(data)
                new_data['instruction'], new_data['fake_dataset'] = make_false_negative_sample(new_data['instruction'], new_data['dataset'], dataset)
                output = evaluate(new_data['instruction'], new_data['input'])
                new_data['pred'] = output
                new_dataset.append(new_data)

with open('kappa/source_prompt_v1.json', 'w') as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)