# !pip install transformers

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
parser.add_argument('--model_name', type=str, default='MODEL_NAME')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--trainset_path', type=str, default='')
parser.add_argument('--devset_path', type=str, default='')
parser.add_argument('--res_path', type=str, default='')
opt = parser.parse_args()
# os.makedirs(opt.upload_folder, exist_ok=True)


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


set_seed(1)

# the model we gonna train, base uncased BERT
# check text classification models here: https://huggingface.co/models?filter=text-classification
model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample
max_length = 128
# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
break_types = ["<0>", "<1>", "<2>", "<3>"]
tokenizer.add_tokens(break_types)

trainset_path = opt.trainset_path
devset_path = opt.devset_path


def get_data(path):
    df = pd.read_csv(path)
    texts = df['token'].values.tolist()
    label = df['label'].values.tolist()
    return texts, label


train_texts, train_labels = get_data(trainset_path)
valid_texts, valid_labels = get_data(devset_path)
target_names = [0, 1]
# tokenize the dataset, truncate when passed `max_length`,
# and pad with 0's when less than `max_length`
train_encodings = tokenizer(train_texts,
                            truncation=True,
                            padding=True,
                            max_length=max_length)
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
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)
# load the model and pass to CUDA
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(target_names)).to("cuda")
model.resize_token_embeddings(len(tokenizer))


# compute the metric: accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': recall,
        'f1': f1,
    }


training_args = TrainingArguments(
    output_dir=os.path.join(opt.res_path, 'results'),  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=opt.
    batch_size,  # batch size per device during training
    per_device_eval_batch_size=opt.batch_size,  # batch size for evaluation
    warmup_steps=1000,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=os.path.join(opt.res_path,
                             'log'),  # directory for storing logs
    load_best_model_at_end=
    True,  # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=1000,  # log & save weights each logging_steps
    save_steps=1000,
    evaluation_strategy="steps",  # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=valid_dataset,  # evaluation dataset
    compute_metrics=
    compute_metrics,  # the callback that computes metrics of interest
)
# train the model
trainer.train()
# evaluate the current model after training
trainer.evaluate()
# saving the fine tuned model & tokenizer
model_path = os.path.join(opt.res_path, opt.model_name)
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)