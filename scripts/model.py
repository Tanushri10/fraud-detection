import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_data_sample(file_path='data/creditcard.csv', sample_size=10000):
    df = pd.read_csv(file_path)
    return df.sample(n=sample_size, random_state=42)

def preprocess_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler for later use
    joblib.dump(scaler, 'scripts/scaler.joblib')
    print("✅ Scaler saved to scripts/scaler.joblib")
    
    return X_scaled, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("✅ Model Evaluation")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return clf

def main():
    print("⏳ Loading sample data...")
    df = load_data_sample()  # load 10k random samples
    print(f"✅ Sample data loaded: {df.shape}")

    X, y = preprocess_data(df)
    model = train_model(X, y)

    joblib.dump(model, 'scripts/random_forest_model.joblib')
    print("✅ Model saved to scripts/random_forest_model.joblib")

if __name__ == "__main__":
    main()


