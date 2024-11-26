#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import nltk
import torch
import warnings
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class UnredactorDataset(Dataset):
    def __init__(self, contexts, labels=None, tokenizer=None, max_length=128):
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = labels is not None

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        inputs = self.tokenizer.encode_plus(
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.has_labels:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def preprocess_context(context):
    if not isinstance(context, str):
        context = ''
    redaction_pattern = r'█+'
    context_replaced = re.sub(redaction_pattern, '[REDACTED]', context)
    return context_replaced.strip()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    return {'accuracy': acc, 'f1': f1}

def main():
    data = pd.read_csv('unredactor.tsv', sep='\t')
    data = data.dropna(subset=['context', 'name']).reset_index(drop=True)
    train_data = data[data['split'] == 'training'].reset_index(drop=True)
    val_data = data[data['split'] == 'validation'].reset_index(drop=True)

    train_data['processed_context'] = train_data['context'].apply(preprocess_context)
    val_data['processed_context'] = val_data['context'].apply(preprocess_context)

    label_encoder = LabelEncoder()
    all_names = pd.concat([train_data['name'], val_data['name']]).unique()
    label_encoder.fit(all_names)
    train_data['label'] = label_encoder.transform(train_data['name'])
    val_data['label'] = label_encoder.transform(val_data['name'])

    num_labels = len(label_encoder.classes_)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)

    train_dataset = UnredactorDataset(
        contexts=train_data['processed_context'].tolist(),
        labels=train_data['label'].tolist(),
        tokenizer=tokenizer
    )
    val_dataset = UnredactorDataset(
        contexts=val_data['processed_context'].tolist(),
        labels=val_data['label'].tolist(),
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("Training the BERT model...")
    trainer.train()

    print("Evaluating the model on validation data...")
    eval_results = trainer.evaluate()
    print(f"Validation Results: {eval_results}")

    test_data = pd.read_csv('test.tsv', sep='\t', header=None)
    test_data.columns = ['id', 'context']
    test_data['processed_context'] = test_data['context'].apply(preprocess_context)
    test_dataset = UnredactorDataset(
        contexts=test_data['processed_context'].tolist(),
        tokenizer=tokenizer
    )

    print("Predicting on test data...")
    raw_pred, _, _ = trainer.predict(test_dataset)
    y_test_pred = np.argmax(raw_pred, axis=1)
    test_names_pred = label_encoder.inverse_transform(y_test_pred)

    submission = pd.DataFrame({'id': test_data['id'], 'name': test_names_pred})
    submission.to_csv('submission.tsv', sep='\t', index=False)
    print("submission.tsv has been generated.")

if __name__ == "__main__":
    main()
