# ML Experiment Pipeline

This project implements a machine learning pipeline for training and evaluating models using scikit-learn and MLflow.

## Project Structure

```
ml_experiment/
├── data/
│   ├── raw/                     # Raw dataset storage
│   │   └── iris.csv            # Raw dataset
│   └── processed/              # Processed data directory
│       ├── X_train.csv         # Training features
│       ├── y_train.csv         # Training labels
│       ├── X_test.csv          # Test features
│       └── y_test.csv          # Test labels
├── notebooks/
│   ├── 01-data_exploration.ipynb   # Data exploration notebook
│   ├── 02-model_training.ipynb     # Model training notebook
│   └── 03-hyperparameter_tuning.ipynb # Hyperparameter tuning
├── scripts/
│   ├── preprocess.py           # Data preprocessing script
│   ├── train_model.py          # Model training script
│   ├── tune_model.py           # Hyperparameter tuning script
│   ├── evaluate_model.py       # Model evaluation script
│   └── predict.py              # Prediction script
├── models/
│   ├── pipeline.pkl            # Serialized pipeline
│   └── best_model.pkl          # Best tuned model
├── mlruns/                     # MLflow tracking
├── environment/
│   ├── conda_env.yaml          # Conda environment spec
│   └── requirements.txt        # Pip requirements
└── setup.sh                    # Setup script
```

## Setup

1. Create and activate the environment:
```bash
conda env create -f environment/conda_env.yaml
conda activate ml_pipeline
```

2. Install dependencies:
```bash
pip install -r environment/requirements.txt
```

3. Run the setup script:
```bash
bash setup.sh
```

## Usage

1. Data Exploration:
   - Open and run `notebooks/01-data_exploration.ipynb`

2. Model Training:
   - Run `python scripts/train_model.py`

3. Hyperparameter Tuning:
   - Run `python scripts/tune_model.py`

4. Model Evaluation:
   - Run `python scripts/evaluate_model.py`

5. Making Predictions:
   - Run `python scripts/predict.py`

## MLflow Tracking

MLflow is used to track experiments. To view the tracking UI:
```bash
mlflow ui
```
Then visit `http://localhost:5000`

## License

MIT
