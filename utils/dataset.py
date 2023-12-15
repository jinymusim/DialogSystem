import torch
import os
import json
from torch.utils.data import Dataset
from datasets import load_dataset
from .dialog_utils import Tokens


class DialogDataset(Dataset):
    def __init__(self, dataset_name: str = "chargoddard/rpguild", cache = "./") -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.data = []
        if os.path.isfile(os.path.join(cache, "train.json")):
            self.data =list( json.load(open(os.path.join(cache, "train.json"), 'r')))
        else:
            raw_dataset = load_dataset(dataset_name)
            self.__create_data(raw_dataset)
            json.dump(self.data, open(os.path.join(cache, "train.json"), 'w+'), indent=6)
        
    def __create_data(self, raw_dataset):
        for datum in raw_dataset['train']:
            character = f"{Tokens.CHAR_TOKEN} {datum['char_name']}, Bio: {datum['bio']}"
            context = f"{Tokens.CONTEXT_TOKEN} " 
            usr_input = f"{Tokens.INPUT_TOKEN} "
            for i, cont in enumerate(datum['context']):
                if i + 1 == len(datum['context']):
                    usr_input += cont['text']
                else:
                    context += f"{cont['text']} "
            response = f"{Tokens.RESPONSE_TOKEN} {datum['reply']}"
            self.data.append({
            'input_ids' : f"{character} {context}{usr_input} {response}"
            })
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    @staticmethod
    def collate(batch,  tokenizer):
        tokenized = tokenizer([datum['input_ids'] + tokenizer.eos_token for datum in batch], return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        return {
            "input_ids" : input_ids,
            "labels": input_ids.type(torch.LongTensor),
            "attention_mask" : attention
        }
