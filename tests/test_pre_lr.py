import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lr_rf_model import preprocess_context, extract_features

def test_preprocess_context_with_redaction():
    context = "Wishing you a Merry Christmas ██████ filled with joy."
    features = preprocess_context(context)
    assert features['redaction_length'] == 6
    assert features['redaction_position'] > 0
    assert 'REDACTED' in features['processed_context']
    assert features['word_before'] == 'Christmas'
    assert features['word_after'] == 'filled'

def test_preprocess_context_without_redaction():
    context = "Enjoy the festive season without any redaction."
    features = preprocess_context(context)
    assert features['redaction_length'] == 0
    assert features['redaction_position'] == 0.0
    assert 'REDACTED' not in features['processed_context']
    assert features['word_before'] == ''
    assert features['word_after'] == ''

def test_extract_features():
    row = pd.Series({
        'context': "Santa Claus gave a gift to ██████ tonight.",
        'name': 'Sakshi'
    })
    features = extract_features(row)
    assert isinstance(features, pd.Series)
    assert 'processed_context' in features
    assert 'redaction_length' in features
    assert features['redaction_length'] == 6
    assert 'word_before' in features
    assert 'word_after' in features
    assert features['word_before'] == 'to'
    assert features['word_after'] == 'tonight'

def test_extract_features_no_redaction():
    row = pd.Series({
        'context': "Let it snow without any redaction.",
        'name': 'Mabel'
    })
    features = extract_features(row)
    assert features['redaction_length'] == 0
    assert features['word_before'] == ''
    assert features['word_after'] == ''

def test_preprocessing_pipeline():
    data = pd.DataFrame({
        'context': [
            "Wishing you a Merry Christmas ██████ filled with joy.",
            "Santa's workshop is busy ████ this year you might not get a gift.",
            "Enjoy the festive season without any redaction."
        ],
        'name': ['Mabel', 'Sage', 'Charles']
    })
    data['name_encoded'] = LabelEncoder().fit_transform(data['name'])
    features = data.apply(extract_features, axis=1)
    data = pd.concat([data, features], axis=1)
    data = data.fillna({'processed_context': '', 'word_before': '', 'word_after': ''})
    X = data[['processed_context', 'word_before', 'word_after', 'redaction_length', 'redaction_position']]
    y = data['name_encoded']

    text_transformer = TfidfVectorizer()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'processed_context'),
            ('cat', categorical_transformer, ['word_before', 'word_after']),
            ('num', numerical_transformer, ['redaction_length', 'redaction_position'])
        ]
    )

    preprocessor.fit(X)
    X_transformed = preprocessor.transform(X)
    assert X_transformed.shape[0] == X.shape[0]

def test_logistic_regression_training():
    data = pd.DataFrame({
        'context': [
            "Wishing you a Merry Christmas ██████ filled with joy.",
            "Santa's workshop is busy ████ this year; you might not get a gift.",
            "Enjoy the festive season without any redaction."
        ],
        'name': ['Mabel', 'Sage', 'Charles']
    })
    label_encoder = LabelEncoder()
    data['name_encoded'] = label_encoder.fit_transform(data['name'])
    features = data.apply(extract_features, axis=1)
    data = pd.concat([data, features], axis=1)
    data = data.fillna({'processed_context': '', 'word_before': '', 'word_after': ''})
    X = data[['processed_context', 'word_before', 'word_after', 'redaction_length', 'redaction_position']]
    y = data['name_encoded']

    text_transformer = TfidfVectorizer()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'processed_context'),
            ('cat', categorical_transformer, ['word_before', 'word_after']),
            ('num', numerical_transformer, ['redaction_length', 'redaction_position'])
        ]
    )

    pipeline_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=2000))
    ])

    pipeline_lr.fit(X, y)
    y_pred = pipeline_lr.predict(X)
    assert len(y_pred) == len(y)
    assert set(y_pred).issubset(set(y))

    try:
        pipeline_lr.predict(X)
    except NotFittedError:
        pytest.fail("Pipeline is not fitted.")

def test_random_forest_training():
    data = pd.DataFrame({
        'context': [
            "Wishing you a Merry Christmas ██████ filled with joy.",
            "Santa's workshop is busy ████ this year; you might not get a gift.",
            "Enjoy the festive season without any redaction."
        ],
        'name': ['Mabel', 'Sage', 'Charles']
    })
    label_encoder = LabelEncoder()
    data['name_encoded'] = label_encoder.fit_transform(data['name'])
    features = data.apply(extract_features, axis=1)
    data = pd.concat([data, features], axis=1)
    data = data.fillna({'processed_context': '', 'word_before': '', 'word_after': ''})
    X = data[['processed_context', 'word_before', 'word_after', 'redaction_length', 'redaction_position']]
    y = data['name_encoded']

    text_transformer = TfidfVectorizer()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, 'processed_context'),
            ('cat', categorical_transformer, ['word_before', 'word_after']),
            ('num', numerical_transformer, ['redaction_length', 'redaction_position'])
        ]
    )

    pipeline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=10))
    ])

    pipeline_rf.fit(X, y)
    y_pred = pipeline_rf.predict(X)
    assert len(y_pred) == len(y)
    assert set(y_pred).issubset(set(y))

    try:
        pipeline_rf.predict(X)
    except NotFittedError:
        pytest.fail("Pipeline is not fitted.")
