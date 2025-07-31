import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data

def main():
    df = load_data('data/creditcard.csv')

    print("✅ Dataset loaded successfully!")
    print(df.head())

if __name__ == "__main__":
    main()

