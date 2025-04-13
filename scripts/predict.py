"""
Script to load a trained model and make predictions.
"""
import joblib
import pandas as pd
import numpy as np

def load_model(model_path='../models/best_model.pkl'):
    """Load the trained model."""
    return joblib.load(model_path)

def predict(model, data):
    """Make predictions using the trained model."""
    return model.predict(data)

def main():
    """Main function to demonstrate prediction."""
    # Load the model
    model = load_model()
    
    # Example: Make predictions on sample data
    # You can modify this to accept input from different sources
    sample_data = pd.DataFrame({
        'sepal length (cm)': [5.1],
        'sepal width (cm)': [3.5],
        'petal length (cm)': [1.4],
        'petal width (cm)': [0.2]
    })
    
    # Make prediction
    prediction = predict(model, sample_data)
    
    print("Sample prediction:", prediction)
    
    return prediction

if __name__ == '__main__':
    main()