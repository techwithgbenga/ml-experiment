"""
Model training script with MLflow tracking.
"""
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

def load_processed_data():
    """Load the processed training data."""
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
    return X_train, y_train

def create_pipeline():
    """Create the ML pipeline."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', random_state=42))
    ])

def train_model():
    """Train the model and log with MLflow."""
    # Load data
    X_train, y_train = load_processed_data()
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Create and train pipeline
        pipeline = create_pipeline()
        pipeline.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("model_type", "SVC")
        mlflow.log_param("kernel", "rbf")
        
        # Save the model
        joblib.dump(pipeline, '../models/pipeline.pkl')
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"Model trained and saved! Run ID: {run.info.run_id}")

if __name__ == '__main__':
    train_model()