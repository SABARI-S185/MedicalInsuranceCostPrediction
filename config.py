"""
Configuration file for Insurance Cost Prediction Project
All parameters and paths are centralized here - DO NOT HARDCODE
"""

# Dataset Configuration
PATHS = {
    'data': 'data/insurance.csv',
    'models': 'models/',
    'plots': 'static/plots/'
}

# Model Training Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'models': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        },
        'ridge': {
            'alpha': 1.0
        },
        'lasso': {
            'alpha': 1.0
        }
    }
}

# Feature Configuration
FEATURES = {
    'numerical': ['age', 'bmi', 'children'],
    'categorical': ['sex', 'smoker', 'region'],
    'target': 'charges',
    'engineered': ['smoker_bmi', 'smoker_age', 'age_bmi']
}

# Input Validation Rules
VALIDATION = {
    'age': {'min': 18, 'max': 100},
    'bmi': {'min': 15.0, 'max': 50.0},
    'children': {'min': 0, 'max': 5}
}

# Model Artifacts
ARTIFACTS = {
    'model': 'models/insurance_model.pkl',
    'scaler': 'models/scaler.pkl',
    'encoders': 'models/encoders.pkl',
    'feature_names': 'models/feature_names.pkl',
    'metadata': 'models/model_metadata.pkl'
}

# Currency Conversion (USD to INR)
USD_TO_INR = 83.0  # 1 USD = 83 INR (approximate rate)

