import sys
import os
import pandas as pd

# Add parent to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data

def run_eda():
    df = load_data('data/creditcard.csv')

    print("✅ EDA Starting...")
    print("Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nClass Distribution:\n", df['Class'].value_counts(normalize=True))

if __name__ == "__main__":
    run_eda()

