from vllm import LLM, SamplingParams
from data.data_utils import WholeLoader

import torch
import csv, json, yaml, random
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_path', type=str, default='test.yaml')
parser.add_argument('--model', type=str, default='ckp/llama13b_source_nickname/ckp_epoch4')
args = parser.parse_args()

yaml_path = args.yaml_path
BASE_MODEL = args.model

with open(yaml_path) as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)

llm = LLM(model=BASE_MODEL, dtype='float16', tensor_parallel_size=torch.cuda.device_count())

data_path = '../../data/IE_INSTRUCTIONS'

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
origin_dataset = origin_dataloader.get_dataset(['test'], shuffle=False)

def make_false_negative_sample(instruction, origin_dataset):
    if origin_dataset in dataset_config['NER']:
        new_dataset = random.choice(dataset_config['NER'])
    else:
        new_dataset = random.choice(dataset_config['RE'])
    return instruction.replace(origin_dataset, new_dataset), new_dataset

def make_nickname_sample(instruction, origin_dataset):
    nickname = origin_dataset[::-1]
    return instruction.replace(origin_dataset, nickname), nickname

for data in tqdm(origin_dataset):
    #print(data)
    data['instruction'], data['fake_dataset'] = make_false_negative_sample(data['instruction'], data['dataset'])
    #data['instruction'], data['nickname'] = make_nickname_sample(data['instruction'], data['dataset'])
    output = evaluate(data['instruction'], data['input'])
    data['pred'] = output
    #break

with open('result/llama13b_fake_source_1.json', 'w') as f:
    json.dump(origin_dataset, f, ensure_ascii=False, indent=4)