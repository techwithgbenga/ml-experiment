"""
Model evaluation script.
"""
import mlflow
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

def load_test_data():
    """Load the test data."""
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv').squeeze()
    return X_test, y_test

def evaluate_model(model_path='../models/best_model.pkl'):
    """Evaluate the model and log metrics."""
    # Load test data
    X_test, y_test = load_test_data()
    
    # Load model
    model = joblib.load(model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Log metrics with MLflow
    with mlflow.start_run() as run:
        mlflow.log_metric("test_accuracy", accuracy)
        
        # Log the classification report as a text artifact
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print(f"Run ID: {run.info.run_id}")

if __name__ == '__main__':
    evaluate_model()