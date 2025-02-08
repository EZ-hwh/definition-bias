import torch
from peft import PeftModel
import re
import csv, json
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, AutoConfig, GenerationConfig
from tqdm import tqdm
from data.data_utils import WholeLoader

BASE_MODEL = "ckp/llama7b_source/ckp_epoch4"
data_path = '../../data/IE_INSTRUCTIONS'
yaml_path = 'train.yaml'
tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Generate on %s" %device)
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map = 'auto',
    )

template_wo_example = 'Instruction:{0}\n\nInput:{1}\nOutput:'

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

def evaluate(
    instruction,
    text=None,
    temperature=0.7,
    top_p=0.95,
    top_k=100,
    repetition_penalty=1,
    length_penalty=1,
    num_beams=4,
    **kwargs,
):
    prompt = template_wo_example.format(instruction, text)
    inputs = tokenizer(prompt, return_tensors="pt")
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
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512
        )
    torch.cuda.empty_cache()
    s = generation_output.sequences[0]
    output = tokenizer.decode(s[len(input_ids[0]):]).split('<end>')[0]
    return output

origin_dataloader = WholeLoader(data_path, yaml_path)
origin_dataset = origin_dataloader.get_dataset(['test'])

for data in tqdm(origin_dataset):
    print(data)
    print(evaluate(data['instruction'], data['input']))
    #break