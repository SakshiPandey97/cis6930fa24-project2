<<<<<<< HEAD
# Unredactor Project-2

## Introduction
This project aims to develop models capable of unredacting names in text documents. The primary challenge is to predict the original names that have been redacted in given contexts. Two approaches were implemented:

1. Traditional Machine Learning Models: Logistic Regression (LR) and Random Forest (RF).
2. Deep Learning Model: BERT (Bidirectional Encoder Representations from Transformers).

Due to the nature of the task and dataset limitations, several challenges were encountered, including class imbalance, lack of additional data, and computational constraints.

## Implementation Details
### 1. Logistic Regression and Random Forest Models
**Script:** `lr_rf_model.py`

**Data Preprocessing**
- **Context Preprocessing**:
  - Replaced redacted portions, the █, with the token 'REDACTED'.
  - Cleaned the text by:
    - Converting to lowercase.
    - Removing punctuation.
    - Removing stop words.
    - Applying stemming using NLTK's SnowballStemmer.

- **Feature Extraction**:
  - **Text Features**: Used TF-IDF Vectorizer with character n-grams (1 to 3 characters) to capture sub-word information.
  - **Categorical Features**: One-hot encoded the words immediately before and after the redaction.
  - **Numerical Features**: Included `redaction_length` and `redaction_position` as features.

- **Handling Class Imbalance**:
  - Applied `class_weight='balanced'` in both LR and RF models to adjust weights inversely proportional to class frequencies.

**Model Training**
- **Logistic Regression**:
  - Used `LogisticRegression` with `solver='saga'` for handling large datasets.
  - Performed hyperparameter tuning using `GridSearchCV` with parameters:
    - Regularization strength (`C`): [0.1, 1.0, 10.0]
    - Penalty: 'l2'

- **Random Forest**:
  - Used `RandomForestClassifier`.
  - Performed hyperparameter tuning using `GridSearchCV` with parameters:
    - Number of estimators (`n_estimators`): [100, 200]
    - Maximum depth (`max_depth`): [None, 10, 20]
    - Minimum samples split (`min_samples_split`): [2, 5]

**Evaluation**
- Evaluated models using the macro-averaged F1 score due to class imbalance.
- Generated classification reports to assess precision, recall, and F1-score for each class.

### 2. BERT Model
**Script:** `bert_model.py`

**Data Preprocessing**
- **Context Preprocessing**:
  - Replaced redacted portions with the token '[REDACTED]' to maintain consistency with BERT's expected input format.

- **Label Encoding**:
  - Encoded names using `LabelEncoder` to convert them into numerical labels for classification.

**Model Training**
- **Model Architecture**:
  - Used `BertForSequenceClassification` pre-trained on the 'bert-base-uncased' model with the number of labels equal to the unique names in the dataset.

- **Training Parameters**:
  - Number of epochs: 3
  - Batch size: 16 for training, 32 for evaluation
  - Optimizer and scheduler handled internally by `Trainer`.

**Evaluation**
- Computed accuracy and weighted F1 score on the validation set.
- Used the Hugging Face `Trainer` API for training and evaluation.

## Results and Errors
### Logistic Regression and Random Forest Models
- **Classification Reports**:
  - Both models showed low precision and recall for most classes, indicating difficulty in predicting the correct names.
  - The models tended to predict the most frequent classes (e.g., 'Sadako') due to class imbalance.

- **Observations**:
  - The Random Forest model occasionally performed better on certain classes but still struggled overall.
  - Logistic Regression sometimes failed to converge within the maximum iterations, indicating the need for further tuning or alternative solvers.

### BERT Model
- **Issue Encountered**:
  - The BERT model predominantly predicted a single name ('Sadako') for all redactions in the test set, suggesting overfitting to the most frequent class or poor generalization due to limited data.

- **Possible Reasons**:
  - **Data Imbalance**: The training data had a significant imbalance in class distribution.
  - **Limited Training Data**: BERT models require substantial amounts of data to fine-tune effectively.
  - **Overfitting**: The model might have overfitted to the training data, failing to generalize to unseen examples.

## Challenges and Limitations
### Difficulty in Obtaining Additional Data
- **Limited Dataset**: Insufficient examples for each class made it challenging for models to learn diverse name representations. Many names had very few samples, which is problematic for classification tasks.

- **Custom Name Redactor**:
  - Developed a custom script (`name_redactor.py`) to redact names from additional text data to generate more training examples. Used spaCy's Named Entity Recognition (NER) to identify and replace names with redaction symbols.
  - **Challenges**: Processing the large dataset was computationally expensive, leading to long processing times and high memory consumption.


## Conclusion
This project highlighted the complexities of unredacting names using both traditional machine learning models and deep learning approaches. Key challenges included:

- **Class Imbalance**: A significant skew in name distribution affected the models' ability to predict less frequent names.
- **Limited Data**: Insufficient training examples hindered the learning process.
- **Computational Limitations**: Processing large datasets was not feasible within the project timeframe.

Despite these challenges, the project provided valuable insights:
- **Model Performance**: Both LR and RF models offered some variability in predictions, though overall accuracy remained low.
- **BERT Limitations**: Fine-tuning BERT without sufficient data led to suboptimal performance.

### Future Work
- **More Data**: I have written a script to redact names from the ImDB dataset however was not able to obtain all the data due to limited time.

## Files Included
- `lr_rf_model.py`: Script for Logistic Regression and Random Forest models with improved preprocessing and feature engineering.
- `bert_model.py`: Script for the BERT model for sequence classification.
- `name_redactor.py`: Custom script used to redact names from text data to generate additional training examples (not fully utilized due to time constraints).
- `submission.tsv`: Output file containing model predictions on the test set.

## How to Run
### Install Dependencies
```bash
pipenv install
```

### Prepare Data
Ensure `unredactor.tsv` and `test.tsv` are placed in the working directory.

### Run Models
**Logistic Regression and Random Forest**:
```bash
pipenv run python lr_rf_model.py
```

**BERT Model**:
```bash
pipenv run python bert_model.py
```

### View Results
- Check the console output for classification reports.
- The `submission.tsv` file will contain the predictions on the test set.
=======
# cis6930fa24project2
>>>>>>> 2cd61ac5d8f34ee297d3cd2d4427fde179c11093
