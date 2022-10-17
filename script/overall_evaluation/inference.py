import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertModel, BertPreTrainedModel
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import argparse
import os
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn
from sklearn.utils import shuffle
from typing import Optional, Tuple, Union
import json

class BertForScoring(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 3
        self.config = config
        # print(config)
        self.bert = BertModel(config)
        # print(self.bert)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        #self.bert.classifier = nn.Linear(config.hidden_size, 3) #config.num_labels)
        self.classifier_new = nn.Linear(config.hidden_size, 3)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier_new(pooled_output)
        # print(logits.size())
        # print(labels.size())
        loss = None
        # if labels is not None:
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             loss = loss_fct(logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # elif self.config.problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=str, default='0')
parser.add_argument('--model_path', type=str, default='../../train/fine-tuning_res/coarse-grain-scoring_')
parser.add_argument('--report', type=bool, default=False)
# parser.add_argument('--testset_path', type=str, default='../dataset/merged_dataset/test_set.csv')
opt = parser.parse_args()
# os.makedirs(opt.upload_folder, exist_ok=True)

# the model we gonna train, base uncased BERT
# check text classification models here: https://huggingface.co/models?filter=text-classification
# model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample
max_length = 128
target_names = [0, 1, 2]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = opt.model_path+opt.idx
# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForScoring.from_pretrained(model_path, num_labels=len(target_names)).to(device)
testset_path = f'../../test/coarse_grain_set/coarse_grain_testset_{opt.idx}.csv'


def get_data(path):
    df = pd.read_csv(path)
    texts = df['token'].values.tolist()
    label = [i-1 for i in df['label'].values.tolist()]
    return texts, label

# train: 1600; test: 400
# train_texts, train_labels = get_data(trainset_path)
valid_texts, valid_labels = get_data(testset_path)
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

def eval():      
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
            res_labels.extend(labels.cpu()[0].numpy().tolist())
            res_preds.extend(preds.cpu().numpy().tolist())
    # if not opt.report:
    #     return {
    #         'accuracy': accuracy_score(res_labels, res_preds),
    #         'precision': precision_score(res_labels, res_preds, average='weighted'),
    #         'recall': recall_score(res_labels, res_preds, average='weighted'),
    #         'f1': f1_score(res_labels, res_preds, average='weighted'),
    #     }
    # else:
    return classification_report(res_labels, res_preds, labels=[0, 1, 2], output_dict=True)

res = eval()
# print(res)
id = opt.model_path[23]
path = f'res{id}.json'
if os.path.exists(path):
    with open(path,'r') as load_f:
        res_list = json.load(load_f)
        res_list.append(res)
        # print(type(res_list), res_list)
    with open(path,'w') as f:
        json.dump(res_list, f)

else:
    with open(path,'w') as f:
        json.dump([res], f)