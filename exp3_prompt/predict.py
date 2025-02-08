import json
import urllib.request
from client import test_api_server
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='test.jsonl')
parser.add_argument('--output_file', type=str, default='llama70b_zs.jsonl')
args = parser.parse_args()

with open(args.input_file, 'r') as f:
    dataset = []
    for line in f.readlines():
        dataset.append(json.loads(line))

with open(args.output_file, 'w') as f:
    for index, data in tqdm(enumerate(dataset)):
        data['output'] = test_api_server(data['items'][0]['content'], [])
        f.write(json.dumps(data) + '\n')