# scripts/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/creditcard.csv')

# Basic overview
print("Top Rows:")
print(df.head())

print("\nClass Distribution:")
print(df['Class'].value_counts())

# Plot class distribution
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (0 = Not Fraud, 1 = Fraud)")
plt.savefig("outputs/figures/class_distribution.png")
