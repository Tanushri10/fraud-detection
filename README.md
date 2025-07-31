Credit Card Fraud Detection
This project explores fraud detection using machine learning techniques on real-world credit card transaction data. It walks through essential steps such as data preprocessing, visualization, model training, and evaluation.

📊 Dataset Overview
Source: Kaggle - Credit Card Fraud Detection

Size: 284,807 transactions

Fraud Cases: Only 492 (highly imbalanced)

Features: Anonymized PCA components, Time, Amount, and Class (target)

🔍 What We Did
1. Data Processing
Cleaned data by removing duplicates and missing values

Scaled important features like Time and Amount to normalize their ranges

Split the dataset while maintaining class proportions using stratified sampling

2. Visualization
Plotted class distribution to highlight imbalance

Explored transaction patterns by Time and Amount for fraud vs. non-fraud

Used a heatmap to analyze correlations between all features

3. Model Building
Built a classification model using Random Forest

Focused on performance metrics suited for imbalanced data (Precision, Recall, F1-score)

Saved the trained model and scaler for reuse

✅ Outcomes
The final model provides a solid baseline for detecting fraudulent transactions in an imbalanced setting. This project demonstrates the importance of preprocessing and thoughtful evaluation when working with sensitive, real-world datasets like financial fraud.

