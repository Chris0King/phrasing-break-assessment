import pandas as pd
import torch

#import spacy
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
import torch.nn.functional as F
import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import argparse
import pandas as pd
import numpy as np
import os
import json
from transformers import BertTokenizerFast

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=str, default='0')
opt = parser.parse_args()
id = opt.idx
max_length = 128

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

testset_path = f"../../test/coarse_grain_set/coarse_grain_testset_{id}.csv"
model_path = "../../train/Break-Bert"
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def get_data(path):
    df = pd.read_csv(path)
    texts = df['token'].values.tolist()
    label = [i - 1 for i in df['label'].values.tolist()]
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


class_num = 3
embed_size = 300  # how big is each word vector
max_features = 120000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 750  # max number of words in a question to use
batch_size = 4  # how many samples to process at once
n_epochs = 5  # how many times to iterate over all samples
n_splits = 5  # Number of K-fold Splits
SEED = 10
debug = 0


class BiLSTM(nn.Module):

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        n_classes = class_num
        self.embedding = nn.Embedding(max_features, embed_size)
        # self.embedding.weight = nn.Parameter(
        #     torch.tensor(embedding_matrix, dtype=torch.float32))
        # self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size,
                            self.hidden_size,
                            bidirectional=True,
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 4, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        #rint(x.size())
        h_embedding = self.embedding(x)
        #_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out


valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)
model = torch.load(f'./new_bilstm_model_{id}')
model.eval()


def eval():
    res_labels = []
    res_preds = []

    with torch.no_grad():
        for data in valid_dataset:
            input_ids = data['input_ids'].unsqueeze(0).to(device)
            labels = data['labels'].unsqueeze(0).to(device)

            # Compute logits
            with torch.no_grad():
                outputs = model(input_ids).detach()
                # outputs = model(input_ids,
                #                 attention_mask=attention_mask,
                #                 labels=labels)
                logits = F.softmax(outputs).cpu().numpy()

            preds = logits.argmax(-1)
            res_labels.extend(labels[0].tolist())
            res_preds.extend(preds.tolist())
    # if not opt.report:
    #     return {
    #         'accuracy': accuracy_score(res_labels, res_preds),
    #         'precision': precision_score(res_labels, res_preds, average='weighted'),
    #         'recall': recall_score(res_labels, res_preds, average='weighted'),
    #         'f1': f1_score(res_labels, res_preds, average='weighted'),
    #     }
    # else:
    return classification_report(res_labels,
                                 res_preds,
                                 labels=[0, 1, 2],
                                 output_dict=True)


res = eval()
print(res)
path = f'new_overall_bilstm_res.json'
if os.path.exists(path):
    with open(path, 'r') as load_f:
        res_list = json.load(load_f)
        res_list.append(res)
        # print(type(res_list), res_list)
    with open(path, 'w') as f:
        json.dump(res_list, f)

else:
    with open(path, 'w') as f:
        json.dump([res], f)