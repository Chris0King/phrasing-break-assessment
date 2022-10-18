from unittest import result
from regex import P
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, BertPreTrainedModel, BertModel
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import argparse
import os
from datasets import load_dataset
import ast
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import DataCollatorForTokenClassification
import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../../train/fine-tuning_res/fine-grain-scoring_')
parser.add_argument('--idx', type=str, default='0')
parser.add_argument('--report', type=bool, default=False)
opt = parser.parse_args()

# model_path = "../train/Bert-Break"
model_path = opt.model_path + opt.idx
model_path2 = "distilbert-base-uncased"
max_length = 128
tokenizer = AutoTokenizer.from_pretrained(model_path2)
break_types = ["<0>", "<1>", "<2>", "<3>"]
tokenizer.add_tokens(break_types)

testset_path = f'../../test/fine_grain_set/fine_grain_testset_{opt.idx}.csv'
data_files = {"test": testset_path}
dataset = load_dataset("csv", data_files=data_files)

dataset['test'].features['label'] = datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "A",
                                "B",
                                "C",
                            ]
                        )
                    )
label_list = dataset['test'].features['label'].feature.names

def convert_list(sentence):
    sentence = sentence.split('<')
    res_list = [sentence[0]]
    for item in sentence[1:]:
        res_list.append('<' + item[:2])
        res_list.append(item[2:])
    return res_list

def tokenize_and_align_labels(examples):
    tokens_list = []
    for i, token in enumerate(examples["token"]):
        tokens_list.append(convert_list(token))
    examples["tokens"] = tokens_list
    # print(examples["tokens"])
    
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)

    labels = []
    for i, label in enumerate(examples[f"label"]):
        label = ast.literal_eval(label)
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    # print(tokenized_inputs)
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns('label')
# dataloader = DataLoader(tokenized_dataset, batch_size=32)

def remove_word(labels):
    break_labels = []
    if len(labels.shape) == 3:
        dim0, dim1, dim2 = labels.shape
    else:
        dim0, dim1 = labels.shape
    for i in range(dim0):
        break_label = []
        for j in range(dim1):
            if j%2 == 0:
                break_label.append(labels[i][j])
        break_labels.append(torch.stack(break_label))
    return torch.stack(break_labels)

class BertForDiagnosis(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier_new = nn.Linear(config.hidden_size, self.num_labels)

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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier_new(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            break_preds = remove_word(logits)
            break_labels = remove_word(labels)
            loss = loss_fct(break_preds.view(-1, self.num_labels), break_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = BertForDiagnosis.from_pretrained(model_path, num_labels=4).to(device)
model.resize_token_embeddings(len(tokenizer))

# compute the metric: accuracy
def compute_metrics(predictions, labels):
    true_predictions = [
      [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
      [label_list[l] for (p,l) in zip(prediction, label) if l != -100]
      for prediction, label in zip(predictions, labels)
    ]
    # 只取break位置计算metric
    true_predictions = [l[1::2] for l in true_predictions]
    true_labels = [l[1::2] for l in true_labels]

    labels = []
    preds = []
    for i, predicted in enumerate(true_predictions):
        labels.extend(true_labels[i])
        preds.extend(predicted)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1': f1_score(labels, preds, average='weighted'),
    }

def eval():      
    res_labels = []
    res_preds = []
  
    with torch.no_grad(): 
        for batch in tokenized_dataset['test']: 
            b_input_ids = torch.Tensor(batch['input_ids'])
            b_input_mask = torch.Tensor(batch['attention_mask'])
            b_labels = torch.Tensor(batch['labels'])
            b_input_ids = b_input_ids.type(torch.LongTensor).unsqueeze(0)
            b_input_mask = b_input_mask.type(torch.LongTensor).unsqueeze(0)
            b_labels = b_labels.type(torch.LongTensor).unsqueeze(0)

            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)
            
            # Compute logits
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask,
                            labels=b_labels)
                logits = outputs.logits
            # predictions = torch.argmax(logits, axis=2)
            preds = torch.argmax(logits, axis=2)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(preds, b_labels)
            ]
            true_labels = [
                [label_list[l] for (p,l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(preds, b_labels)
            ]
            # 只取break位置计算metric
            true_predictions = [l[1::2] for l in true_predictions]
            true_labels = [l[1::2] for l in true_labels]
            for i, predicted in enumerate(true_predictions):
                res_labels.extend(true_labels[i])
                res_preds.extend(predicted)

    # res_labels = MultiLabelBinarizer().fit_transform(res_labels)
    # res_preds = MultiLabelBinarizer().fit_transform(res_preds)
    # if not opt.report:
    # return {
    #     'accuracy': accuracy_score(res_labels, res_preds),
    #     'precision': precision_score(res_labels, res_preds, average='weighted'),
    #     'recall': recall_score(res_labels, res_preds, average='weighted'),
    #     'f1': f1_score(res_labels, res_preds, average='weighted'),
    # }
    # else:
    return classification_report(res_labels, res_preds, labels=['A', "B", "C"], output_dict=True)


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