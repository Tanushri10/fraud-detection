import pandas as pd

# Load the dataset
data = pd.read_csv('data/creditcard.csv')
print(data.head())

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Class', data=data)
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.show()

plt.figure(figsize=(10,5))

sns.histplot(data[data['Class'] == 0]['Amount'], bins=50, color='green', label='Non-Fraud', kde=True)
sns.histplot(data[data['Class'] == 1]['Amount'], bins=50, color='red', label='Fraud', kde=True)

plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))

sns.histplot(data[data['Class'] == 0]['Time'], bins=50, color='green', label='Non-Fraud', kde=True)
sns.histplot(data[data['Class'] == 1]['Time'], bins=50, color='red', label='Fraud', kde=True)

plt.title('Transaction Time Distribution')
plt.xlabel('Time (seconds)')
plt.legend()
plt.show()

plt.figure(figsize=(15,12))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
