import torch
from torch.utils.data import Dataset

def data_transform(dataframe, tokenizer, model_name):
    if model_name == 'albert':
        map = {'neutral': 1, 'contradiction':2, 'entailment':0}

    if model_name == 'xlnet':
        map = {'neutral': 1, 'contradiction':2, 'entailment':0}

    if model_name == 'deberta':
        map = {'neutral': 1, 'contradiction':0, 'entailment':2}

    if model_name == 'roberta':
        map = {'neutral': 1, 'contradiction':0, 'entailment':2}

    dataframe = dataframe[dataframe['gold_label'].apply(lambda x: x in ['neutral', 'contradiction', 'entailment'])]
    labels = dataframe['gold_label'].apply(lambda x: map[x])

    encoded_inputs = tokenizer(list(dataframe['sentence1']), list(dataframe['sentence2']), padding=True, truncation=True, max_length=60, return_tensors='pt')
    
    return encoded_inputs, torch.tensor(labels, dtype=torch.long)

class BertDataset(Dataset):
    def __init__(self, encoded_inputs, labels):
        self.input_ids = encoded_inputs['input_ids']
        self.mask = encoded_inputs['attention_mask']
        self.labels = labels

    def __getitem__(self, item):
        ids   = self.input_ids[item]
        mask  = self.mask[item]
        label = self.labels[item]
        
        return {"input_ids": ids, "attention_mask": mask, "label": label}

    def __len__(self):
        return self.labels.shape[0]