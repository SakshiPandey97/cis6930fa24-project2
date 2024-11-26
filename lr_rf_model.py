#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def preprocess_context(context):
    if not isinstance(context, str):
        context = ''
    redaction_pattern = r'█+'
    match = re.search(redaction_pattern, context)
    if match:
        redaction_start = match.start()
        redaction_end = match.end()
        redaction_length = redaction_end - redaction_start
        redaction_position = redaction_start / len(context) if len(context) > 0 else 0.0
        context_replaced = re.sub(redaction_pattern, ' REDACTED ', context)
        tokens = word_tokenize(context_replaced)
        try:
            redacted_index = tokens.index('REDACTED')
            word_before = tokens[redacted_index - 1] if redacted_index > 0 else ''
            word_after = tokens[redacted_index + 1] if redacted_index < len(tokens) - 1 else ''
        except ValueError:
            word_before = ''
            word_after = ''
        processed_context = context_replaced.strip()
        return {
            'processed_context': processed_context if processed_context else '',
            'redaction_length': redaction_length,
            'redaction_position': redaction_position,
            'word_before': word_before,
            'word_after': word_after,
        }
    else:
        processed_context = context.strip()
        return {
            'processed_context': processed_context if processed_context else '',
            'redaction_length': 0,
            'redaction_position': 0.0,
            'word_before': '',
            'word_after': '',
        }

def extract_features(row):
    context = row.get('context', '')
    features = preprocess_context(context)
    return pd.Series(features)

def main():
    data = pd.read_csv('unredactor.tsv', sep='\t')
    data = data.dropna(subset=['context']).reset_index(drop=True)
    train_data = data[data['split'] == 'training'].reset_index(drop=True)
    val_data = data[data['split'] == 'validation'].reset_index(drop=True)
    train_features = train_data.apply(extract_features, axis=1)
    train_data = pd.concat([train_data, train_features], axis=1)
    val_features = val_data.apply(extract_features, axis=1)
    val_data = pd.concat([val_data, val_features], axis=1)
    train_data = train_data.fillna({'processed_context': '', 'word_before': '', 'word_after': ''})
    val_data = val_data.fillna({'processed_context': '', 'word_before': '', 'word_after': ''})
    print("Number of NaN values in 'processed_context' in training data:", train_data['processed_context'].isna().sum())
    print("Number of NaN values in 'processed_context' in validation data:", val_data['processed_context'].isna().sum())
    print("Number of NaN values in 'context' in the original data:", data['context'].isna().sum())
    N = 100
    name_counts = train_data['name'].value_counts()
    top_names = name_counts.nlargest(N).index.tolist()
    train_data_top = train_data[train_data['name'].isin(top_names)].reset_index(drop=True)
    val_data_top = val_data[val_data['name'].isin(top_names)].reset_index(drop=True)
    X_train = train_data_top[['processed_context', 'word_before', 'word_after', 'redaction_length', 'redaction_position']].copy()
    y_train = train_data_top['name']
    X_val = val_data_top[['processed_context', 'word_before', 'word_after', 'redaction_length', 'redaction_position']].copy()
    y_val = val_data_top['name']
    X_train.loc[:, 'processed_context'] = X_train['processed_context'].astype(str)
    X_val.loc[:, 'processed_context'] = X_val['processed_context'].astype(str)
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000))
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'processed_context'),
            ('cat', categorical_transformer, ['word_before', 'word_after']),
            ('num', numerical_transformer, ['redaction_length', 'redaction_position'])
        ])
    pipeline_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    print("Training Logistic Regression...")
    pipeline_lr.fit(X_train, y_train)
    y_pred_lr = pipeline_lr.predict(X_val)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_val, y_pred_lr, zero_division=0))
    print("Training Random Forest...")
    pipeline_rf.fit(X_train, y_train)
    y_pred_rf = pipeline_rf.predict(X_val)
    print("Random Forest Classification Report:")
    print(classification_report(y_val, y_pred_rf, zero_division=0))
    print("Starting GridSearchCV for Logistic Regression...")
    param_grid_lr = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['saga'],
    }
    grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search_lr.fit(X_train, y_train)
    print("Best parameters for Logistic Regression:")
    print(grid_search_lr.best_params_)
    y_pred_lr_gs = grid_search_lr.predict(X_val)
    print("Logistic Regression Classification Report after GridSearch:")
    print(classification_report(y_val, y_pred_lr_gs, zero_division=0))
    print("Starting GridSearchCV for Random Forest...")
    param_grid_rf = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
    }
    grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    print("Best parameters for Random Forest:")
    print(grid_search_rf.best_params_)
    y_pred_rf_gs = grid_search_rf.predict(X_val)
    print("Random Forest Classification Report after GridSearch:")
    print(classification_report(y_val, y_pred_rf_gs, zero_division=0))

    test_data = pd.read_csv('test.tsv', sep='\t', header=None)
    test_data.columns = ['id', 'context']
    test_features = test_data.apply(extract_features, axis=1)
    test_data = pd.concat([test_data, test_features], axis=1)
    test_data = test_data.fillna({'processed_context': '', 'word_before': '', 'word_after': ''})
    X_test = test_data[['processed_context', 'word_before', 'word_after', 'redaction_length', 'redaction_position']].copy()
    X_test.loc[:, 'processed_context'] = X_test['processed_context'].astype(str)
    print("Predicting on test data using the best model...")
    
    y_test_pred = grid_search_rf.predict(X_test)
    submission = pd.DataFrame({'id': test_data['id'], 'name': y_test_pred})
    submission.to_csv('submission.tsv', sep='\t', index=False)
    print("submission.tsv has been generated.")

if __name__ == "__main__":
    main()