"""
Hyperparameter tuning script using GridSearchCV.
"""
import mlflow
import pandas as pd
from sklearn.model_selection import GridSearchCV
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
        ('classifier', SVC(random_state=42))
    ])

def tune_hyperparameters():
    """Perform hyperparameter tuning with GridSearchCV."""
    # Load data
    X_train, y_train = load_processed_data()
    
    # Define parameter grid
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['rbf', 'linear'],
        'classifier__gamma': ['scale', 'auto', 0.1, 1],
    }
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Perform grid search
        grid_search.fit(X_train, y_train)
        
        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Save the best model
        joblib.dump(grid_search.best_estimator_, '../models/best_model.pkl')
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == '__main__':
    tune_hyperparameters()