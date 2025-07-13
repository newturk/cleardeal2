# Author: Shubham Kumar | Gmail: shubhamkumar831015@gmail.com | Contact: +91 9508741536 | GitHub: https://github.com/newturk/cleardeal2
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---
# Professional Model Training Script for Open Source Use
# - Trains a GradientBoostingClassifier on leads_dataset.csv
# - Handles preprocessing, evaluation, and model saving
# ---

DATA_PATH = 'data/leads_dataset.csv'
MODEL_PATH = 'model/model.pkl'

# Features to use (no PII, only open data)
FEATURES = [
    'age', 'job', 'marital', 'education', 'default', 'balance',
    'housing', 'loan', 'contact', 'duration', 'campaign', 'pdays',
    'previous', 'poutcome', 'comments'  # 'comments' can be dropped for ML, used for LLM re-ranker
]
TARGET = 'intent_score'

# Load data
df = pd.read_csv(DATA_PATH)

# Drop 'comments' and 'consent' for ML model (used in re-ranker and compliance)
X = df.drop(['intent_score', 'comments', 'consent', 'y'], axis=1)
y = df['intent_score'] // 100  # Convert to binary 0/1 for classification

# Identify categorical and numeric features
categorical = X.select_dtypes(include=['object']).columns.tolist()
numeric = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numeric)
])

# Model pipeline
pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', GradientBoostingClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print('Model Evaluation:')
print(classification_report(y_test, y_pred, target_names=['Low Intent', 'High Intent']))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Save model
joblib.dump(pipeline, MODEL_PATH)
print(f'Model saved to {MODEL_PATH}') 