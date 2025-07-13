# Author: Shubham Kumar | Gmail: shubhamkumar831015@gmail.com | Contact: +91 9508741536 | GitHub: https://github.com/newturk/cleardeal2
import pandas as pd
import numpy as np
import random

# ---
# Professional Preprocessing Script for Open Source Use
# - Loads the UCI Bank Marketing dataset
# - Adds synthetic 'comments' and 'consent' fields
# - Maps 'y' to a 0/100 intent score
# - Saves as /data/leads_dataset.csv
# ---

# Path to original dataset
INPUT_PATH = 'dataset/bank/bank.csv'
OUTPUT_PATH = 'data/leads_dataset.csv'

# Synthetic comments pool for demo/LLM re-ranker
demo_comments = [
    'Interested in more details',
    'Please call back',
    'Not interested',
    'Urgent inquiry',
    'Requested follow up',
    'Needs more information',
    'Considering options',
    'Asked about rates',
    'Wants to speak to manager',
    'No response after call',
    'Positive feedback',
    'Negative feedback',
    'Asked for email brochure',
    'Will decide next week',
    'Not available',
]

def add_synthetic_fields(df):
    # Add synthetic comments
    df['comments'] = np.random.choice(demo_comments, size=len(df))
    # Add synthetic consent (always True for training/demo)
    df['consent'] = True
    return df

def map_target(df):
    # Map 'y' to intent score (0/100)
    df['intent_score'] = df['y'].map({'yes': 100, 'no': 0})
    return df

def main():
    # Read the dataset (semicolon separator)
    df = pd.read_csv(INPUT_PATH, sep=';')
    # Add synthetic fields
    df = add_synthetic_fields(df)
    # Map target
    df = map_target(df)
    # Save processed dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Processed dataset saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main() 