Credit Card Fraud Detection
This project aims to detect fraudulent transactions using machine learning techniques on a highly imbalanced dataset. It demonstrates data preprocessing, visualization, model building, evaluation, and saving the model for future use.

📁 Project Structure
bash
Copy
Edit
creditcard-fraud-detection/
├── data/
│   └── creditcard.csv           # Raw dataset
├── model/
│   ├── model.py                 # Script to train and save the model
│   ├── predict.py               # Script to load model and predict new inputs
│   ├── fraud_model.pkl          # Saved trained model
│   └── scaler.pkl               # Saved MinMaxScaler
├── visualization/
│   └── visualize.py             # Data visualization and exploration
├── README.md                    # Project documentation
📊 Dataset
Source: Kaggle Credit Card Fraud Detection Dataset

Description: The dataset contains transactions made by European cardholders in September 2013. It has 284,807 transactions with 492 frauds (0.172%).

Features: Time, Amount, 28 anonymized PCA features (V1 to V28), and Class (target: 0 = Not Fraud, 1 = Fraud)

⚙️ Technologies Used
Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Joblib

🧼 Data Preprocessing
Dropped duplicate entries

Normalized Amount and Time using MinMaxScaler

Stratified train-test split (80:20) to preserve fraud ratio

Stored the scaler as scaler.pkl for deployment

📊 Data Visualization
Using matplotlib and seaborn, we visualized:

Class Imbalance

Transaction Amounts and Time (fraud vs non-fraud)

Correlation heatmap of features

These plots helped identify the data distribution and relationships among features.

🤖 Model Building
Algorithm: Random Forest Classifier

Trained on preprocessed data to classify transactions

Model was saved as fraud_model.pkl using joblib

Evaluation metrics: Confusion matrix, precision, recall, f1-score


📈 Results
The Random Forest model performs well on the imbalanced dataset, especially in precision and recall for the fraud class.

Visualization and processing steps make the model interpretable and production-ready.

✅ Conclusion
This project highlights the complete pipeline of solving a real-world fraud detection problem using supervised learning—from data preprocessing and EDA to model training and deployment preparation. Future enhancements can include:

Anomaly detection techniques

Real-time fraud detection with streaming data

Model explainability (e.g., SHAP, LIME)

