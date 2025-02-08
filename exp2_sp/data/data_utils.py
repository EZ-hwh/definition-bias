import os, sys, json, yaml
from data.prompt import *
from tqdm import tqdm
import random

class BaseLoader:
    def __init__(self, data_path, split, use_nickname = False):
        self.data_name = data_path.split('/')[-1]
        self.nickname = self.data_name[::-1]
        with open(os.path.join(data_path, 'labels.json')) as f:
            self.labels = json.load(f)
        self.dataset = []
        self.use_nickname = use_nickname
        
        if 'train' in split:
            #print('Load training set...')
            with open(os.path.join(data_path, 'train.json')) as f:
                self.dataset += json.load(f)
        if 'test' in split:
            #print('Load test set...')
            with open(os.path.join(data_path, 'test.json')) as f:
                self.dataset += json.load(f)
        if 'val' in split:
            #print('Load validation set...')
            with open(os.path.join(data_path, 'dev.json')) as f:
                self.dataset += json.load(f)

        self.format_instruction()

    def format_instruction(self):
        raise NotImplementedError()

    def get_dataset(self, max_samples=None, shuffle=True):
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
            for label in self.labels:
                if self.nickname and random.random() > 0.5:
                    data_name = self.nickname
                else:
                    data_name = self.data_name
                entities = [entity for entity in data['entities'] if entity['type'] == label]
                if not entities:
                    continue
                new_data = {
                    'dataset': self.data_name,
                    'instruction': NER_prompt.format(data_name, label),
                    'input': data['sentence'],
                    'output': self.serialize_output(entities)
                }
                new_dataset.append(new_data)
        self.dataset = new_dataset

    def serialize_output(self, entities):
        new_entities = [f"{item['type']}: {item['name']}" for item in entities]
        return '; '.join(new_entities)

class RELoader(BaseLoader):
    def format_instruction(self):
        new_dataset = []
        for data in self.dataset:
            for label in self.labels:
                if self.nickname and random.random() > 0.5:
                    data_name = self.nickname
                else:
                    data_name = self.data_name
                relations = [relation for relation in data['relations'] if relation['type'] == label]
                if not relations:
                    continue
                new_data = {
                    'dataset': self.data_name,
                    'instruction': RE_prompt.format(data_name, label),
                    'input': data['sentence'],
                    'output': self.serialize_output(relations)
                }
                new_dataset.append(new_data)
        self.dataset = new_dataset

    def serialize_output(self, relations):
        new_relations = [f"({item['head']['name']}, {item['type']}, {item['tail']['name']})" for item in relations]
        return ', '.join(new_relations)


class WholeLoader:
    def __init__(self, data_path, yaml_path):
        self.data_path = data_path
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.dataset_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(self.dataset_config)

    def get_dataset(self, data_split=['train'], use_nickname=False, shuffle=True):
        whole_dataset = []
        for task in self.dataset_config.keys():
            for dataset in self.dataset_config[task]:
                if task == 'NER':
                    dataloader = NERLoader(os.path.join(self.data_path, task, dataset), data_split, use_nickname=use_nickname)
                elif task == 'RE':
                    dataloader = RELoader(os.path.join(self.data_path, task, dataset), data_split, use_nickname=use_nickname)
                inst_dataset = dataloader.get_dataset(shuffle=shuffle)
                print(f'{task}\t{dataset}:\t{len(inst_dataset)}')
                #print(len(inst_dataset))
                whole_dataset.extend(inst_dataset)
        print('='*50)
        print(f'Whole dataset:',len(whole_dataset))
        return whole_dataset

if __name__ == '__main__':
    dataloader = WholeLoader(data_path='../../../data/IE_INSTRUCTIONS',yaml_path='../train.yaml')
    dataset = dataloader.get_dataset()
    print(dataset[0])