#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p data/{raw,processed}
mkdir -p models
mkdir -p mlruns

# Download the Iris dataset if it doesn't exist
if [ ! -f data/raw/iris.csv ]; then
    python -c "from sklearn.datasets import load_iris; import pandas as pd; iris = load_iris(); df = pd.DataFrame(iris.data, columns=iris.feature_names); df['target'] = iris.target; df.to_csv('data/raw/iris.csv', index=False)"
fi

# Set up MLflow
export MLFLOW_TRACKING_URI="mlruns"

echo "Setup completed successfully!"