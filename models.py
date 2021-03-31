from transformers import AlbertForSequenceClassification, RobertaForSequenceClassification, XLNetForSequenceClassification, DebertaV2ForSequenceClassification
from transformers import AlbertTokenizer, RobertaTokenizer, XLNetTokenizer, DebertaV2Tokenizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
import shutil
import pickle
import wget
import os

from utils import data_transform, BertDataset

def get_model(model_name='albert'):
    # currently available models are 'roberta', 'albert', 'xlnet' и 'deberta'

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name=='albert':
        files = ['config.json', 'pytorch_model.bin', 'spiece.model', 'special_tokens_map.json', 'tokenizer_config.json']
        base_url = 'https://huggingface.co/ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli/resolve/main/'
        os.mkdir('albert')

        for file in files:
            url = base_url + file
            wget.download(url, out='albert/' + file)

        model     = AlbertForSequenceClassification.from_pretrained('albert', num_labels=3).to(device)
        tokenizer = AlbertTokenizer.from_pretrained('albert')
        shutil.rmtree('albert')

    if model_name == 'roberta':
        model     = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli', num_labels=3).to(device)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')

    if model_name == 'xlnet':
        files = ['config.json', 'pytorch_model.bin', 'spiece.model', 'special_tokens_map.json', 'tokenizer_config.json']
        base_url = 'https://huggingface.co/ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli/resolve/main/'
        os.mkdir('xlnet')

        for file in files:
            url = base_url + file
            wget.download(url, out='xlnet/' + file)

        model     = XLNetForSequenceClassification.from_pretrained('xlnet', num_labels=3).to(device)
        tokenizer = XLNetTokenizer.from_pretrained('xlnet')
        shutil.rmtree('xlnet')

    if model_name == 'deberta':
        model     = DebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v2-xxlarge-mnli', num_labels=3).to(device)
        tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v2-xxlarge-mnli')

    return model, tokenizer

def get_dataloader(dataframe, tokenizer, model_name, batch_size=16):
    X_val, y_val = data_transform(dataframe, tokenizer, model_name)
    return DataLoader(BertDataset(X_val, y_val),  batch_size=batch_size, shuffle=False)

def evaluation(model, dataloader, model_name, criterion=nn.CrossEntropyLoss(), return_softmax=False, save_preds=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = []
    ground_true = []
    
    num_iter = 0
    model.eval()
    with torch.no_grad():
        correct = 0
        num_objs = 0
        for batch in tqdm(dataloader):
            ids, mask, labels = (batch[inp].to(device) for inp in ["input_ids", "attention_mask", "label"])

            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            correct += (outputs.logits.argmax(-1)==labels.squeeze()).sum()

            preds.append(outputs.logits.clone().detach().cpu())
            ground_true.append(labels)

            num_objs += labels.shape[0]
            num_iter += 1
        print(f"Accuracy: {correct/num_objs}")

    sm_preds = []
    gold_label = []
    sm = nn.Softmax(dim=1)

    for pred in preds:
        sm_preds.extend([sm(el.unsqueeze(0)).squeeze(0) for el in pred])

    for label in ground_true:
        gold_label.extend([el for el in label])

    if return_softmax==False:
        preds = [pred.argmax(-1) for pred in sm_preds]
    else:
        preds = sm_preds

    output = {"preds": preds, 'gold_label': gold_label}
    with open(model_name + '.pickle', 'wb') as f:
        pickle.dump(output, f)

    return {"preds": preds, 'gold_label': gold_label}