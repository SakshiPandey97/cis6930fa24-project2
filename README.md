# Unredactor Project-2

## Introduction
This project aims to develop models capable of unredacting names in text documents. The primary challenge is to predict the original names that have been redacted in given contexts. Two approaches were implemented:

1. **Traditional Machine Learning Models**: Logistic Regression (LR) and Random Forest (RF).
2. **Deep Learning Model**: BERT (Bidirectional Encoder Representations from Transformers).

Due to the nature of the task and dataset limitations, several challenges were encountered, including class imbalance, lack of additional data, and computational constraints.

## Implementation Details

### 1. Logistic Regression and Random Forest Models
**Script**: `lr_rf_model.py`

#### Data Preprocessing

- **Context Preprocessing**:
  - Replaced redacted portions, the `█`, with the token `'REDACTED'`.
  - Extracted features such as the words immediately before and after the redaction, redaction length, and redaction position.

- **Handling Class Imbalance**:
  - Limited the training data to the top 100 most frequent names to ensure sufficient examples for each class.

#### Feature Extraction

- **Text Features**: Used TF-IDF Vectorizer with word n-grams (1 to 2 words) to capture contextual information.
- **Categorical Features**: One-hot encoded the words immediately before and after the redaction.
- **Numerical Features**: Included `redaction_length` and `redaction_position` as features.

#### Model Training

- **Logistic Regression**:
  - Used `LogisticRegression` with default solver.
  - Performed hyperparameter tuning using `GridSearchCV` with parameters:
    - Regularization strength (`C`): [0.1, 1.0, 10.0]
    - Penalty: `'l2'`
    - Solver: `'saga'`

- **Random Forest**:
  - Used `RandomForestClassifier`.
  - Performed hyperparameter tuning using `GridSearchCV` with parameters:
    - Number of estimators (`n_estimators`): [100, 200]
    - Maximum depth (`max_depth`): [None, 10, 20]

#### Evaluation

- Evaluated models using the macro-averaged F1 score.
- Generated classification reports to assess precision, recall, and F1-score for each class.

### 2. BERT Model
**Script**: `bert_model.py`

#### Data Preprocessing

- **Context Preprocessing**:
  - Replaced redacted portions with the token `'[REDACTED]'` to maintain consistency with BERT's expected input format.

- **Label Encoding**:
  - Encoded names using `LabelEncoder` to convert them into numerical labels for classification.

#### Model Training

- **Model Architecture**:
  - Used `BertForSequenceClassification` pre-trained on the `'bert-base-uncased'` model with the number of labels equal to the unique names in the dataset.

- **Training Parameters**:
  - Number of epochs: 3
  - Batch size: 16 for training, 32 for evaluation
  - Optimizer and scheduler handled internally by `Trainer`.

#### Evaluation

- Computed accuracy and weighted F1 score on the validation set.
- Used the Hugging Face `Trainer` API for training and evaluation.

## Results and Errors

### Logistic Regression and Random Forest Models
- **Classification Reports**:
  - Both models showed low precision and recall for most classes, indicating difficulty in predicting the correct names.
  - The models tended to predict the most frequent classes due to class imbalance.

- **Observations**:
  - Limiting the dataset to the top 100 names improved the representation of each class but did not significantly enhance model performance.
  - Additional features or advanced preprocessing might be needed to improve results.

### BERT Model
- **Issue Encountered**:
  - The BERT model predominantly predicted a single name (e.g., 'Sadako') for all redactions in the test set, suggesting overfitting to the most frequent class or poor generalization due to limited data.

- **Possible Reasons**:
  - **Data Imbalance**: Significant imbalance in class distribution.
  - **Limited Training Data**: BERT models require substantial amounts of data to fine-tune effectively.
  - **Overfitting**: The model might have overfitted to the training data, failing to generalize to unseen examples.

## Challenges and Limitations

- **Difficulty in Obtaining Additional Data**
  - **Limited Dataset**: Insufficient examples for each class made it challenging for models to learn diverse name representations. Many names had very few samples, which is problematic for classification tasks.

- **Custom Name Redactor**:
  - Developed a custom script (`name_redactor.py`) to redact names from additional text data to generate more training examples. Used spaCy's Named Entity Recognition (NER) to identify and replace names with redaction symbols.
  - **Challenges**: Processing the large dataset was computationally expensive, leading to long processing times and high memory consumption. Therefore, I was unable to use this.

## Conclusion
This project highlighted the complexities of unredacting names using both traditional machine learning models and deep learning approaches. Key challenges included:

- **Class Imbalance**: A significant skew in name distribution affected the models' ability to predict less frequent names.
- **Limited Data**: Insufficient training examples hindered the learning process.
- **Computational Limitations**: Processing large datasets was not feasible within the project timeframe.

Despite these challenges, the project provided valuable insights:

- **Model Performance**: Both LR and RF models offered some variability in predictions, though overall accuracy remained low.
- **BERT Limitations**: Fine-tuning BERT without sufficient data led to suboptimal performance.

## Future Work
- **More Data**: I have written a script to redact names from the IMDb dataset; however, I was not able to obtain all the data due to limited time.

## Files Included
- `lr_rf_model.py`: Script for Logistic Regression and Random Forest models.
- `bert_model.py`: Script for the BERT model for sequence classification.
- `test_lr_rf_model.py`: Unit tests for the `lr_rf_model.py` script.
- `submission.tsv`: Output file containing model predictions on the test set.

## How to Run
### Install Dependencies
```bash
pipenv install
```

### Prepare Data
Ensure `unredactor.tsv` and `test.tsv` are placed in the working directory.

### Run Models
- **Logistic Regression and Random Forest**:
  ```bash
  pipenv run python lr_rf_model.py
  ```

- **BERT Model**:
  ```bash
  pipenv run python bert_model.py
  ```

### Run Unit Tests
To run the unit tests for the `lr_rf_model.py` script:
```bash
pipenv run pytest test_lr_rf_model.py
```

### View Results
Check the console output for classification reports and test results. The `submission.tsv` file contains the predictions on the test set.

