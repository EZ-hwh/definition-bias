import os, sys, json, yaml
from data.prompt import *
from tqdm import tqdm
import random

few_shot_template = 'Input:{0}\nOutput:{1}'

class BaseLoader:
    def __init__(self, data_path, split, few_shot=0):
        self.data_name = data_path.split('/')[-1]
        with open(os.path.join(data_path, 'labels.json')) as f:
            self.labels = json.load(f)
        self.dataset = []

        self.few_shot = few_shot
        if few_shot > 0:
            with open(os.path.join(data_path, 'train.json')) as f:
                self.few_shot_dataset = json.load(f)

        if 'train' in split:
            with open(os.path.join(data_path, 'train.json')) as f:
                self.dataset += json.load(f)
        if 'test' in split:
            with open(os.path.join(data_path, 'test.json')) as f:
                self.dataset += json.load(f)
        if 'valid' in split:
            with open(os.path.join(data_path, 'dev.json')) as f:
                self.dataset += json.load(f)

        self.format_instruction()

    def format_instruction(self):
        raise NotImplementedError()

    def get_dataset(self, max_samples=None, shuffle=False):
        if shuffle:
            random.shuffle(self.dataset)
        if max_samples:
            return self.dataset[:max_samples]
        else:
            return self.dataset

class NERLoader(BaseLoader):
    def format_instruction(self):
        new_dataset = []
        for data in self.dataset:
            new_data = {
                'dataset_name': self.data_name,
                'instruction': NER_prompt.format(str(self.labels)),
                'few_shot_case': '\n\n'.join([few_shot_template.format(d['sentence'], self.serialize_output(d['entities'])) for d in random.sample(self.few_shot_dataset, self.few_shot)]) if self.few_shot > 0 else '',
                'label_list': [item['type'] for item in data['entities']],
                'input': data['sentence'],
                'output': self.serialize_output(data['entities'])
            }
            new_dataset.append(new_data)
        self.dataset = new_dataset

    def serialize_output(self, entities):
        new_entities = [f"{item['type']}: {item['name']}" for item in entities]
        if new_entities:
            return '; '.join(new_entities)
        else:
            return 'Sorry, There\'s no result'

class RELoader(BaseLoader):
    def format_instruction(self):
        new_dataset = []
        for data in self.dataset:
            new_data = {
                'dataset_name': self.data_name,
                'instruction': RE_prompt.format(str(self.labels)),
                'few_shot_case': '\n\n'.join([few_shot_template.format(d['sentence'], self.serialize_output(d['relations'])) for d in random.sample(self.few_shot_dataset, self.few_shot)]) if self.few_shot > 0 else '',
                'label_list': [item['type'] for item in data['relations']],
                'input': data['sentence'],
                'output': self.serialize_output(data['relations'])
            }
            new_dataset.append(new_data)
        self.dataset = new_dataset

    def serialize_output(self, relations):
        new_relations = [f"({item['head']['name']}, {item['type']}, {item['tail']['name']})" for item in relations]
        if new_relations:
            return ', '.join(new_relations)
        else:
            return 'Sorry, There\'s no result'



class WholeLoader:
    def __init__(self, data_path, yaml_path):
        self.data_path = data_path
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(self.dataset_config)

    def get_dataset(self, data_split=['train'], few_shot=0, max_samples_per_dataset=None):
        whole_dataset = []
        for task in self.dataset_config.keys():
            for dataset in self.dataset_config[task]:
                if task == 'NER':
                    dataloader = NERLoader(os.path.join(self.data_path, task, dataset), data_split, few_shot)
                elif task == 'RE':
                    dataloader = RELoader(os.path.join(self.data_path, task, dataset), data_split, few_shot)
                inst_dataset = dataloader.get_dataset(shuffle=True, max_samples=max_samples_per_dataset)
                print(f'{task}\t{dataset}:\t{len(inst_dataset)}')
                whole_dataset.extend(inst_dataset)
        print('='*50)
        print(f'Whole dataset:',len(whole_dataset))
        return whole_dataset

if __name__ == '__main__':
    dataloader = WholeLoader(data_path='../../../data/IE_INSTRUCTIONS',yaml_path='../train.yaml')
    dataset = dataloader.get_dataset()
    print(dataset[0])