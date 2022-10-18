import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast
import os
import json
import random
from transformers.file_utils import is_tf_available, is_torch_available
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

embed_size = 128  # how big is each word vector
max_features = 30526  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 750  # max number of words in a question to use
batch_size = 8  # how many samples to process at once
n_epochs = 4  # how many times to iterate over all samples
n_splits = 5  # Number of K-fold Splits
SEED = 10
debug = 0
class_num = 3

parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=str, default='0')
opt = parser.parse_args()
idx = opt.idx


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)


set_seed(SEED)


def get_data(path):
    df = pd.read_csv(path)
    texts = df['token'].values.tolist()
    label = [i - 1 for i in df['label'].values.tolist()]
    return texts, label


trainset_path = f'/home/zy/break-solution/test/coarse_grain_set/coarse_grain_trainset_{idx}.csv'
devset_path = f'/home/zy/break-solution/test/coarse_grain_set/coarse_grain_devset_{idx}.csv'
all_text, all_labels = get_data(
    '/home/zy/break-solution/test/coarse_grain_res.csv')
train_texts, train_labels = get_data(trainset_path)
valid_texts, valid_labels = get_data(devset_path)

max_length = 128
model_path = "../../train/Break-Bert"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
train_encodings = tokenizer(train_texts,
                            truncation=True,
                            padding='max_length',
                            max_length=max_length)
valid_encodings = tokenizer(valid_texts,
                            truncation=True,
                            padding='max_length',
                            max_length=max_length)
all_encodings = tokenizer(all_text,
                          truncation=True,
                          padding='max_length',
                          max_length=max_length)
# x_train,x_test,y_train,y_test = train_test_split(all_encodings,np.asarray(all_labels))

train_X = train_encodings['input_ids']
train_y = train_labels
test_X = valid_encodings['input_ids']
test_y = valid_labels


class BiLSTM(nn.Module):

    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 1024
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


print(idx)
model = BiLSTM()
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                    model.parameters()),
                             lr=0.001)
print(model)
model.cuda()

# Load train and test in CUDA Memory
x_train = torch.tensor(train_X, dtype=torch.long).cuda()
y_train = torch.tensor(train_y, dtype=torch.long).cuda()
x_cv = torch.tensor(test_X, dtype=torch.long).cuda()
y_cv = torch.tensor(test_y, dtype=torch.long).cuda()

# Create Torch datasets
train = torch.utils.data.TensorDataset(x_train, y_train)
valid = torch.utils.data.TensorDataset(x_cv, y_cv)

# Create Data Loaders
train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=batch_size,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid,
                                           batch_size=batch_size,
                                           shuffle=False)

train_loss = []
valid_loss = []
best_loss = 30000

for epoch in range(n_epochs):
    start_time = time.time()
    # Set model to train configuration
    model.train()
    avg_loss = 0.
    for i, (x_batch, y_batch) in enumerate(train_loader):
        # Predict/Forward Pass
        y_pred = model(x_batch)
        # Compute loss
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
    # Set model to validation configuration -Doesn't get trained here
    model.eval()
    avg_val_loss = 0.
    val_preds = np.zeros((len(x_cv), class_num))

    for i, (x_batch, y_batch) in enumerate(valid_loader):
        y_pred = model(x_batch).detach()
        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
        # keep/store predictions
        val_preds[i * batch_size:(i + 1) *
                  batch_size] = F.softmax(y_pred).cpu().numpy()

    # Check Accuracy
    val_accuracy = sum(val_preds.argmax(axis=1) == test_y) / len(test_y)
    train_loss.append(avg_loss)
    valid_loss.append(avg_val_loss)
    elapsed_time = time.time() - start_time
    if avg_val_loss < best_loss:
        torch.save(model, f'new_bilstm_model_{idx}')
        best_loss = avg_val_loss
    print(
        'Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f}  \t val_acc={:.4f}  \t time={:.2f}s'
        .format(epoch + 1, n_epochs, avg_loss, avg_val_loss, val_accuracy,
                elapsed_time))
