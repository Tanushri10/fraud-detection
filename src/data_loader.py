import pandas as pd
import os

def load_data(filename='creditcard.csv'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found.")
    df = pd.read_csv(filename)
    return df
