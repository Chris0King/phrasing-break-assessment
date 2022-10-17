from transformers import BertForSequenceClassification, BertTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# import warnings
# warnings.filterwarnings("ignore")


# model_path = "../train/2classification-bert-base-uncased"
# max_length = 512
# target_names = [0, 1]

# model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(target_names)).to("cuda")
# tokenizer = BertTokenizerFast.from_pretrained(model_path)

# def get_prediction(text):
#     # prepare our text into tokenized sequence
#     inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
#     # perform inference to our model
#     outputs = model(**inputs)
#     print(outputs)
#     # get output probabilities by doing softmax
#     probs = outputs[0].softmax(1)
#     # executing argmax function to get the candidate label
#     return target_names[probs.argmax()]

# # Example #1
# text = "by<0>that<0>made<3>in<0>america<3>which<0>is<0>the<0>worst<1>conceivable"
# print(get_prediction(text))
# # Example #2
# text = "by<0>that<0>made<3>in<0>america<3>which<0>is<0>the<0>worst<1>conceivable"
# print(get_prediction(text))


import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import argparse
import os
import torch
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../train/pre-training_res/Break-BERT-2')
parser.add_argument('--testset_path', type=str, default='../dataset/merged_dataset/test_set.csv')
opt = parser.parse_args()
# os.makedirs(opt.upload_folder, exist_ok=True)

# the model we gonna train, base uncased BERT
# check text classification models here: https://huggingface.co/models?filter=text-classification
# model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample
max_length = 128
# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(opt.model_path, do_lower_case=True)
# trainset_path = opt.trainset_path
testset_path = opt.testset_path


def get_data(path):
    df = pd.read_csv(path)
    texts = df['token'].values.tolist()
    label = df['label'].values.tolist()
    return texts, label

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# train: 1600; test: 400
# train_texts, train_labels = get_data(trainset_path)
valid_texts, valid_labels = get_data(testset_path)
target_names = [0, 1]
# tokenize the dataset, truncate when passed `max_length`,
# and pad with 0's when less than `max_length`
valid_encodings = tokenizer(valid_texts,
                            truncation=True,
                            padding=True,
                            max_length=max_length)


class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


# convert our tokenized data into a torch Dataset
# train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)
# load the model and pass to CUDA
model = BertForSequenceClassification.from_pretrained(
    opt.model_path, num_labels=len(target_names)).to(device)


def test_species():      
    acc = []
    precision = []
    recall = []
    res_f1 = []
    res_labels = []
    res_preds = []
  
    with torch.no_grad(): 
        for data in valid_dataset: 
            input_ids = data['input_ids'].unsqueeze(0).to(device)
            attention_mask = data['attention_mask'].unsqueeze(0).to(device)
            labels = data['labels'].unsqueeze(0).to(device)
            
            # Compute logits
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
                logits = outputs.logits
            
            preds = logits.argmax(-1)
            res_labels.extend(labels.cpu()[0])
            res_preds.extend(preds.cpu())
             
    return {
        'accuracy': accuracy_score(res_labels, res_preds),
        'precision': precision_score(res_labels, res_preds),
        'recall': recall_score(res_labels, res_preds),
        'f1': f1_score(res_labels, res_preds),
    }

print(test_species())

