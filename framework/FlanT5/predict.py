from data.data_utils import WholeLoader

import torch
import csv, json, yaml, random, os
from transformers import T5ForConditionalGeneration, T5Tokenizer, LlamaTokenizer, AutoConfig, GenerationConfig
from peft import get_peft_model, PeftModel
from tqdm import tqdm

data_path = '/mnt/data122/datasets/Information Extraction/academic_dataset/IE_INSTRUCTIONS'
template_wo_example = ' Instruction:{0}\n\nInput:{1}\nOutput:'
yaml_path = 'test.yaml'
#model_name = 'No_tuning'
model_name = 'flan_t5_weighted'
BASE_MODEL = f"ckp/{model_name}/ckp_epoch4"
#BASE_MODEL = '../../../models/llama-2-13b-hf'

tokenizer = T5Tokenizer.from_pretrained(BASE_MODEL)

with open(yaml_path) as f:
    dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Generate on %s" %device)
if device == "cuda":
    model = T5ForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        device_map = 'auto',
    )

def generate_prompt(instruction, text=None):
    return template_wo_example.format(instruction, text)

def evaluate(
    instruction,
    inputs,
    temperature=0.7,
    top_p=0.7,
    top_k=40,
    num_beams=4,
    length_penalty=1.0,
    repetition_penalty=1,
    **kwargs,
):
    prompt = generate_prompt(instruction, inputs)
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split('</s>')[0]

origin_dataloader = WholeLoader(data_path, yaml_path)
origin_dataset = origin_dataloader.get_dataset(['test'])

#origin_dataset = [data for data in origin_dataset if data['dataset_name'] == dataset_name]

for data in tqdm(origin_dataset):
    output = evaluate(data['instruction'], data['input'])
    data['pred'] = output

if not os.path.exists(f'result/{model_name}'):
    os.makedirs(f'result/{model_name}')

with open(f'result/{model_name}/overall.json', 'w') as f:
    json.dump(origin_dataset, f, ensure_ascii=False, indent=4)