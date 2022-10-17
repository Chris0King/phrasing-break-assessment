from typing import Optional, Tuple, Union
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification, BertPreTrainedModel, BertModel
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import argparse
import os
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn

import torch, gc

gc.collect()
torch.cuda.empty_cache()


class BertForScoring(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 3
        self.config = config
        # print(config)
        self.bert = BertModel(config)
        # print(self.bert)
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
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
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


parser = argparse.ArgumentParser()
parser.add_argument('--model_path',
                    type=str,
                    default='../../train0/Break-Bert')
parser.add_argument('--res_path',
                    type=str,
                    default='../../train/fine-tuning_res')
parser.add_argument('--idx', type=str, default='0')
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


set_seed(10)

model_path = opt.model_path
# model_path = "../train/BERT-Break-30w"
# model_path = "bert-base-uncased"
max_length = 128
target_names = [0, 1, 2]

model = BertForScoring.from_pretrained(model_path).to("cuda")
tokenizer = BertTokenizerFast.from_pretrained(model_path)
if model_path.count('/') == 0:
    # print("999")
    break_types = ["<0>", "<1>", "<2>", "<3>"]
    tokenizer.add_tokens(break_types)
    model.resize_token_embeddings(len(tokenizer))


def get_data(path):
    df = pd.read_csv(path)
    # df = shuffle(df)
    texts = df['token'].values.tolist()
    label = [i - 1 for i in df['label'].values.tolist()]
    return texts, label


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


# compute the metric: accuracy
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': recall,
        'f1': f1,
    }


res_path = opt.res_path

training_args = TrainingArguments(
    output_dir=os.path.join(res_path, 'results'),  # output directory
    num_train_epochs=4,  # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,  # batch size for evaluation
    warmup_steps=10,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=os.path.join(res_path, 'log'),  # directory for storing logs
    load_best_model_at_end=
    True,  # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    metric_for_best_model='f1',
    logging_steps=20,  # log & save weights each logging_steps
    save_steps=20,
    evaluation_strategy="steps",  # evaluate each `logging_steps`
)


def train_eval_model(trainset_path, devset_path):
    # train: 1600; test: 400
    train_texts, train_labels = get_data(trainset_path)
    valid_texts, valid_labels = get_data(devset_path)
    # valid_texts, valid_labels = get_data(testset_path)
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
    # convert our tokenized data into a torch Dataset
    train_dataset = NewsGroupsDataset(train_encodings, train_labels)
    valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

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
    model_path = os.path.join(opt.res_path, f"coarse-grain-scoring_{opt.idx}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


idx = 4
trainset_path = f'../../test/coarse_grain_set/coarse_grain_trainset_{opt.idx}.csv'
devset_path = f'../../test/coarse_grain_set/coarse_grain_devset_{opt.idx}.csv'
# testset_path = f'../test/coarse_grain_set/coarse_grain_testset_{opt.idx}.csv'
train_eval_model(trainset_path, devset_path)