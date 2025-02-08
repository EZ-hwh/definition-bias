from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from data.data_utils import WholeLoader

import torch
import csv, json
#from transformers import AutoModelForCausalLM, T5Tokenizer, LlamaTokenizer, AutoConfig, GenerationConfig
from tqdm import tqdm

BASE_MODEL = "ckp/flant5_large_source/ckp_epoch4"
data_path = '../../data/IE_INSTRUCTIONS'
yaml_path = 'train.yaml'
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side='left'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Generate on %s" % device)
if device == "cuda":
    llm = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float16,
        device_map = "auto",
    )


template_wo_example = 'Instruction:{0}\n\nInput:{1}\nOutput:'

def generate_prompt(instruction, text=None):
    return template_wo_example.format(instruction, text)

def evaluate(
    instruction,
    inputs,
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    num_beams=4,
    length_penalty=1,
    repetition_penalty=1,
    **kwargs,
):
    text = generate_prompt(instruction, inputs)
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty = repetition_penalty,
        length_penalty = length_penalty,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = llm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512
        )
    torch.cuda.empty_cache()
    s = generation_output.sequences[0]
    output = tokenizer.decode(s).split('<end>')[0]
    #print(output)
    return output

def batch_evaluate(
    datas,
    temperature=0.7,
    top_p=0.7,
    top_k=40,
    num_beams=1,
    length_penalty=1,
    repetition_penalty=1,
    **kwargs,
):
    text = [generate_prompt(data['instruction'], data['input']) for data in datas]
    #print(tokenizer.batch_decode(text,skip_special_tokens=True))
    inputs = tokenizer(text, return_tensors="pt", padding='longest')
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty = repetition_penalty,
        length_penalty = length_penalty,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = llm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128
        )
    torch.cuda.empty_cache()
    outputs = generation_output.sequences
    output = [tokenizer.decode(s).split('</s>')[0].replace('<pad>','').strip() for s in outputs]
    #print(output)
    return output

origin_dataloader = WholeLoader(data_path, yaml_path)
origin_dataset = origin_dataloader.get_dataset(['test'], shuffle=False)

bs = 4
for index in tqdm(range(len(origin_dataset - 1)//bs + 1)):
    #print(data)
    data = origin_dataset[bs*index:bs*(index+1)]
    outputs = batch_evaluate(data)
    for i in range(bs):
        origin_dataset[bs*index+i]['pred'] = outputs[i]
    
    #break

with open('result/flant5_large_true_source.json', 'w') as f:
    json.dump(origin_dataset, f, ensure_ascii=False, indent=4)