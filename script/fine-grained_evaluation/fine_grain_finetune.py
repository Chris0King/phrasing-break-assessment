from unittest import result
from regex import P
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, BertPreTrainedModel, BertModel
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
import random
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

parser = argparse.ArgumentParser()
parser.add_argument('--res_path',
                    type=str,
                    default='../../train/fine-tuning_res')
parser.add_argument('--model_path', type=str, default='../../train/Break-Bert')
parser.add_argument('--idx', type=str, default='0')
opt = parser.parse_args()


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

# model_path = "../train/Bert-Break"
model_path = opt.model_path
model_path2 = "distilbert-base-uncased"
max_length = 128
target_names = [1, 2, 3]
tokenizer = AutoTokenizer.from_pretrained(model_path2)
break_types = ["<0>", "<1>", "<2>", "<3>"]
tokenizer.add_tokens(break_types)

trainset_path = f'../../test/fine_grain_set/fine_grain_trainset_{opt.idx}.csv'
devset_path = f'../../test/fine_grain_set/fine_grain_devset_{opt.idx}.csv'
res_path = '../../train'

data_files = {"train": trainset_path, "test": devset_path}
dataset = load_dataset("csv", data_files=data_files)
# dataset = dataset.train_test_split(test_size=100, shuffle=False, seed=42)
# print(dataset.keys())
dataset['train'].features['label'] = datasets.Sequence(
    datasets.features.ClassLabel(names=[
        "O",
        "A",
        "B",
        "C",
    ]))
dataset['test'].features['label'] = datasets.Sequence(
    datasets.features.ClassLabel(names=[
        "O",
        "A",
        "B",
        "C",
    ]))
label_list = dataset['test'].features['label'].feature.names
metric = datasets.load_metric("seqeval")


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

    tokenized_inputs = tokenizer(examples["tokens"],
                                 is_split_into_words=True,
                                 truncation=True,
                                 max_length=max_length)

    labels = []
    for i, label in enumerate(examples[f"label"]):
        label = ast.literal_eval(label)
        word_ids = tokenized_inputs.word_ids(
            batch_index=i)  # Map tokens to their respective word.
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

    # labels = []
    # for i, label in enumerate(examples[f"label"]):
    #     label = ast.literal_eval(label)
    #     labels.append(label)

    tokenized_inputs["labels"] = labels
    # print(tokenized_inputs)
    return tokenized_inputs


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
# tokenized_dataset = tokenized_dataset.remove_columns('A-res_form')
# tokenized_dataset = tokenized_dataset.remove_columns('B-res_form')
# tokenized_dataset = tokenized_dataset.remove_columns('C-res_form')
tokenized_dataset = tokenized_dataset.remove_columns('label')
# tokenized_dataset = tokenized_dataset.remove_columns('token')
# print(tokenized_dataset['train'])
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
# exit()


def remove_word(labels):
    break_labels = []
    if len(labels.shape) == 3:
        dim0, dim1, dim2 = labels.shape
    else:
        dim0, dim1 = labels.shape
    for i in range(dim0):
        break_label = []
        for j in range(dim1):
            if j % 2 == 0:
                break_label.append(labels[i][j])
        break_labels.append(torch.stack(break_label))
    return torch.stack(break_labels)


class BertForDiagnosis(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
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
            loss = loss_fct(break_preds.view(-1, self.num_labels),
                            break_labels.view(-1))

        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = BertForDiagnosis.from_pretrained(model_path, num_labels=4).to('cuda')
model.resize_token_embeddings(len(tokenizer))


# compute the metric: accuracy
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[
        label_list[p] for (p, l) in zip(prediction, label) if l != -100
    ] for prediction, label in zip(predictions, labels)]
    true_labels = [[
        label_list[l] for (p, l) in zip(prediction, label) if l != -100
    ] for prediction, label in zip(predictions, labels)]
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
        'precision': precision_score(labels, preds, average='macro'),
        'recall': recall_score(labels, preds, average='macro'),
        'f1': f1_score(labels, preds, average='macro'),
    }


training_args = TrainingArguments(
    output_dir=os.path.join(opt.res_path, 'results'),  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,  # batch size for evaluation
    # learning_rate = 2e-5,
    warmup_steps=20,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir=os.path.join(opt.res_path,
                             'log'),  # directory for storing logs
    load_best_model_at_end=
    True,  # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    # metric_for_best_model='f1',
    logging_steps=20,  # log & save weights each logging_steps
    save_steps=20,
    evaluation_strategy="steps",  # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=tokenized_dataset["train"],  # training dataset
    eval_dataset=tokenized_dataset["test"],  # evaluation dataset
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=
    compute_metrics,  # the callback that computes metrics of interest
)
# train the model
trainer.train()
# evaluate the current model after training
trainer.evaluate()
# saving the fine tuned model & tokenizer
save_path = os.path.join(opt.res_path, f"fine-grain-scoring_{opt.idx}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)