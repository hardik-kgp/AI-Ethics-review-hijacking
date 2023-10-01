#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import io
import random
import argparse
import numpy as np
import mxnet as mx
from tqdm import tqdm
import gluonnlp as nlp
from gluonnlp.calibration import BertLayerCollector
# this notebook assumes that all required scripts are already
# downloaded from the corresponding tutorial webpage on http://gluon-nlp.mxnet.io
from bert import data
np.set_printoptions(precision=3)

# argument parser.
def get_args():
    parser = argparse.ArgumentParser("train/test gloun NLP's implementation of BERT")
    parser.add_argument("-cmp", "--conf_mat_path", default=None, type=str)
    parser.add_argument("-te", "--test", action="store_true")
    parser.add_argument("-tr", "--train", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=100)
    args = parser.parse_args()
    
    return args
# nlp.utils.check_version('0.8.1')
args = get_args()
seed = args.seed
np.random.seed(seed)
random.seed(seed)
mx.random.seed(100*seed)
# change `ctx` to `mx.cpu()` if no GPU is available.
ctx = mx.gpu(0)

bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                            dataset_name='book_corpus_wiki_en_uncased',
                                            pretrained=True, ctx=ctx, use_pooler=True,
                                            use_decoder=False, use_classifier=False)
# print(bert_base)
# print(vocabulary)
bert_classifier = nlp.model.BERTClassifier(bert_base, num_classes=5, dropout=0.1)
# only need to initialize the classifier layer.
bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
bert_classifier.hybridize(static_alloc=True)
# softmax cross entropy loss for classification
# 
loss_function = mx.gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)
train_metric = mx.metric.Accuracy()
test_metric = mx.metric.Accuracy()
# load dataset.
lr = 1e-5
num_discard_samples = 1 # skip the first line, which is the schema
field_separator = nlp.data.Splitter('\t') # split fields by tabs
field_indices = [2, 3, 4] # fields to select from the file
val_filename = "balanced_review_hijack_val.tsv"
test_filename = "balanced_review_hijack_test.tsv"
train_filename = "balanced_review_hijack_train.tsv"
best_model_path = f"best_bert_balanced_w_brands_{seed}_lr_{lr}.params"
data_train_raw = nlp.data.TSVDataset(filename=train_filename,
                                     field_separator=field_separator,
                                     num_discard_samples=num_discard_samples,
                                     field_indices=field_indices)
data_test_raw = nlp.data.TSVDataset(filename=test_filename,
                                    field_separator=field_separator,
                                    num_discard_samples=num_discard_samples,
                                    field_indices=field_indices)
data_val_raw = nlp.data.TSVDataset(filename=val_filename,
                                   field_separator=field_separator,
                                   num_discard_samples=num_discard_samples,
                                   field_indices=field_indices)
# sample_id = 0
# # product title, features and descriptiom
# print(data_train_raw[sample_id][0])
# # review summary (title) and review text
# print(data_train_raw[sample_id][1])
# # class from 0 to 4.
# # class mapping: {'-': 0, "sbsc": 1, "sbdc": 2, "dbsc": 3, "dbdc": 4}
# print(data_train_raw[sample_id][2])


# Use the vocabulary from pre-trained model for tokenization
bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)
# The maximum length of an input sequence
max_len = 250
# The labels for the two classes [(0 = not similar) or  (1 = similar)]
all_labels = [str(i) for i in range(5)]
# whether to transform the data as sentence pairs.
# for single sentence classification, set pair=False
# for regression task, set class_labels=None
# for inference without label available, set has_label=False
pair = True
transform = data.transform.BERTDatasetTransform(bert_tokenizer, max_len,
                                                class_labels=all_labels,
                                                has_label=True,
                                                pad=True,
                                                pair=pair)
data_train = data_train_raw.transform(transform)
data_test = data_test_raw.transform(transform)
data_val = data_val_raw.transform(transform)
# The hyperparameters
batch_size = 16
# The FixedBucketSampler and the DataLoader for making the mini-batches
train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_train],
                                            batch_size=batch_size, shuffle=True)
test_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_test],
                                           batch_size=batch_size, shuffle=False)
val_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in data_val],
                                           batch_size=batch_size, shuffle=False)
train_dataloader = mx.gluon.data.DataLoader(data_train, batch_sampler=train_sampler)
test_dataloader = mx.gluon.data.DataLoader(data_test, batch_sampler=test_sampler)
val_dataloader = mx.gluon.data.DataLoader(data_val, batch_sampler=val_sampler)
trainer = mx.gluon.Trainer(bert_classifier.collect_params(), 'adam',
                           {'learning_rate': lr, 'epsilon': 1e-9})
# Collect all differentiable parameters
# `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
# The gradients for these params are clipped later
params = [p for p in bert_classifier.collect_params().values() if p.grad_req != 'null']
grad_clip = 1
# keep track of class wise accuracy.
class ClassAccuracy:
    def __init__(self, num_classes: int=2):
        self.num_classes = num_classes
        self.reset()
    
    def get(self):
        return (self.class_matches/self.class_counts)
    
    def reset(self):        
        self.class_counts = np.zeros(self.num_classes)
        self.class_matches = np.zeros(self.num_classes)
    
    def update(self, labels: np.ndarray, preds: np.ndarray):
        labels = np.eye(self.num_classes)[labels,:]
        preds = np.eye(self.num_classes)[preds,:]
        self.class_counts += labels.sum(axis=0)
        self.class_matches += (labels*preds).sum(axis=0)
    
# store the confusion matrix
class ConfusionMatrix:
    def __init__(self, num_classes: int=2):
        self.num_classes = num_classes
        self.reset()
    
    def get(self):
        return self.matrix.astype(int)
    
    def reset(self):        
        self.matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, labels: np.ndarray, preds: np.ndarray):
        """each row represents true class. within the row each class represents the preds."""
        labels = np.eye(self.num_classes)[labels,:]
        preds = np.eye(self.num_classes)[preds,:]
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0)
        # print(labels.T.shape, preds.shape)
        self.matrix += (labels.T @ preds)

# Training the model with only three epochs
log_interval = 4
num_epochs = 20
# best_acc = 0
best_valid_loss = 1000
train_class_acc = ClassAccuracy(5)
test_conf_mat = ConfusionMatrix(5)
test_class_acc = ClassAccuracy(5)
# def get_match_count(preds, labels):
#     count = 0
#     for pred, label in zip(preds, labels):
#         count += int(pred == label)
        
#     return count
train_model = args.train
test_model = args.test
if train_model:
    for epoch_id in range(num_epochs):
        step_loss = 0
        train_metric.reset()
        train_class_acc.reset()
        # acc_matches = 0
        # acc_tot = 0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_id, (token_ids, segment_ids, valid_length, label) in pbar:
            with mx.autograd.record():
                # Load the data to the GPU
                token_ids = token_ids.as_in_context(ctx)
                valid_length = valid_length.as_in_context(ctx)
                segment_ids = segment_ids.as_in_context(ctx)
                label = label.as_in_context(ctx)
                # Forward computation
                out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
                loss = loss_function(out, label).mean()
            # And backwards computation
            loss.backward()
            # Gradient clipping
            trainer.allreduce_grads()
            nlp.utils.clip_grad_global_norm(params, 1)
            trainer.update(1)
            step_loss += loss.asscalar()
            # acc_matches += (label.asnumpy().squeeze() == np.argmax(out.asnumpy(), axis=1)).sum()
            # acc_tot += len(label)
            train_metric.update([label], [out])
            train_class_acc.update(
                label.asnumpy().squeeze(), 
                np.argmax(out.asnumpy(), axis=1)
            )
            # Printing vital information
            pbar.set_description(f"{epoch_id}: loss={loss.asscalar():.4f} lr={trainer.learning_rate:.4f} acc={100*train_metric.get()[1]:.2f}")
            
        
        test_class_acc.reset()
        test_metric.reset()
        valid_loss = 0
        step_loss = 0
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

        for batch_id, (token_ids, segment_ids, valid_length, label) in pbar:
            # Load the data to the GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Forward computation
            out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
            loss = loss_function(out, label).mean()
            valid_loss += loss.asscalar()
            test_metric.update([label], [out])
            test_class_acc.update(
                label.asnumpy().squeeze(), 
                np.argmax(out.asnumpy(), axis=1)
            )
            # Print log info
            pbar.set_description(f"{epoch_id}: loss={loss.asscalar():.4f} acc={100*test_metric.get()[1]:.2f}")
        valid_loss /= len(val_dataloader)
        print(f"\x1b[32;1mfinished epoch {epoch_id+1}/{num_epochs} test_acc: {100*test_metric.get()[1]:.2f}% train_acc: {100*train_metric.get()[1]:.2f}%\x1b[0m")
        # if test_metric.get()[1] > best_acc:
        if valid_loss < best_valid_loss:
            # best_acc = test_metric.get()[1]
            best_valid_loss = valid_loss
            print(f"saving best model till now for seed={seed} to {best_model_path}")
            bert_classifier.save_parameters(best_model_path)
            # print(f"\x1b[34;1mbest_acc: {best_acc}\x1b[0m")
            print(f"\x1b[34;1mbest_valid_loss: {best_valid_loss}\x1b[0m")
        print(f"\x1b[32;1mtest_class_wise_acc: {test_class_acc.get()} train_class_wise_acc: {train_class_acc.get()}\x1b[0m")
        
elif test_model:
    print("testing model")
    bert_classifier.load_parameters(best_model_path, ctx=ctx)
    test_class_acc.reset()
    test_conf_mat.reset()
    test_metric.reset()
    test_loss = 0
    step_loss = 0
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for batch_id, (token_ids, segment_ids, valid_length, label) in pbar:
        # Load the data to the GPU
        token_ids = token_ids.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx)
        segment_ids = segment_ids.as_in_context(ctx)
        label = label.as_in_context(ctx)
        # Forward computation
        out = bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
        loss = loss_function(out, label).mean()
        test_loss += loss.asscalar()
        test_metric.update([label], [out])
        test_class_acc.update(
            label.asnumpy().squeeze(), 
            np.argmax(out.asnumpy(), axis=1)
        )
        test_conf_mat.update(
            label.asnumpy().squeeze(), 
            np.argmax(out.asnumpy(), axis=1)
        )
        # Print log info
        pbar.set_description(f"loss={loss.asscalar():.4f} acc={100*test_metric.get()[1]:.2f}")
    test_loss /= len(test_dataloader)
    print(f"test_acc: {100*test_metric.get()[1]:.2f}%")
    print(f"class wise accuracies: {100*test_class_acc.get()}")
    print(f"confusion matrix:\n {test_conf_mat.get()}")
    #         if (batch_id + 1) % (log_interval) == 0:
    #             print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
    #                          .format(epoch_id, batch_id + 1, len(train_dataloader),
    #                                  step_loss / log_interval,
    #                                  trainer.learning_rate, train_metric.get()[1]))
    #             step_loss = 0
    #         if (batch_id + 1) % (log_interval) == 0:
    #             print('[Epoch {} Batch {}/{}] loss={:.4f}, acc={:.3f}'
    #                          .format(epoch_id, batch_id + 1, len(test_dataloader),
    #                                  step_loss/log_interval, test_metric.get()[1]))
    #             step_loss = 0