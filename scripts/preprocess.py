"""
Data preprocessing script for the ML pipeline.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath='../data/raw/iris.csv'):
    """Load the dataset."""
    return pd.read_csv(filepath)

def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess the data and split into train/test sets."""
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test):
    """Save the processed datasets."""
    X_train.to_csv('../data/processed/X_train.csv', index=False)
    X_test.to_csv('../data/processed/X_test.csv', index=False)
    y_train.to_csv('../data/processed/y_train.csv', index=False)
    y_test.to_csv('../data/processed/y_test.csv', index=False)

if __name__ == '__main__':
    # Load the data
    df = load_data()
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Save the processed data
    save_processed_data(X_train, X_test, y_train, y_test)
    
    print("Data preprocessing completed successfully!")