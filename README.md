# 🏥 Insurance Cost Prediction System

A comprehensive Machine Learning web application that predicts annual insurance costs based on individual characteristics. Built with Flask, scikit-learn, and modern web technologies.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements a complete ML pipeline for predicting insurance costs, including:

- **Data Exploration & Analysis**: Comprehensive EDA with visualizations
- **Feature Engineering**: BMI categories, age groups, interaction features
- **Model Training**: 5 different algorithms (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)
- **Model Evaluation**: Cross-validation with multiple metrics (R², RMSE, MAE, MAPE)
- **Web Application**: Flask-based interactive prediction interface
- **REST API**: JSON endpoint for programmatic access

## ✨ Features

### Machine Learning Pipeline
- ✅ Exploratory Data Analysis (EDA) with 7+ visualizations
- ✅ Feature engineering (BMI categories, interaction features)
- ✅ Multiple model comparison (5 algorithms)
- ✅ 5-fold cross-validation
- ✅ Comprehensive evaluation metrics
- ✅ Model persistence and versioning

### Web Application
- ✅ User-friendly prediction form with validation
- ✅ Real-time input validation
- ✅ BMI calculator (optional feature)
- ✅ Detailed cost breakdown
- ✅ Comparison charts and insights
- ✅ EDA dashboard with visualizations
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Modern, professional UI

### Technical Features
- ✅ Configuration-based (no hardcoding)
- ✅ Robust error handling
- ✅ Model artifact persistence
- ✅ RESTful API endpoint
- ✅ Comprehensive testing suite

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone/Download the Repository

```bash
cd "C:\Users\HP\OneDrive\Desktop\ML Project"
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Download the insurance dataset from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
2. Place the `insurance.csv` file in the `data/` directory

**Note:** The dataset should have the following columns:
- `age`: Age of primary beneficiary
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of children covered
- `smoker`: Smoking status (yes/no)
- `region`: Residential area (northeast/northwest/southeast/southwest)
- `charges`: Individual medical costs (target variable)

## 📖 Usage

### Step 1: Train the Model

Before running the web application, train the ML model:

```bash
python train_model.py
```

This script will:
- Load and inspect the dataset
- Perform exploratory data analysis (saves plots to `static/plots/`)
- Engineer features
- Train 5 different models
- Evaluate and compare models
- Select the best model
- Save model artifacts to `models/` directory

**Expected Output:**
- Model performance metrics for all 5 models
- Best model selection
- Saved artifacts: `insurance_model.pkl`, `scaler.pkl`, `encoders.pkl`, `feature_names.pkl`, `model_metadata.pkl`
- Visualization plots in `static/plots/`

### Step 2: Run the Flask Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Step 3: Access the Web Interface

Open your browser and navigate to:
- **Home/Prediction**: `http://localhost:5000/`
- **Analysis Dashboard**: `http://localhost:5000/analysis`
- **API Endpoint**: `http://localhost:5000/api/predict` (POST)

## 📁 Project Structure

```
insurance-prediction/
│
├── app.py                      # Flask web application
├── train_model.py              # ML training script
├── config.py                   # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/
│   └── insurance.csv           # Dataset (not included, download from Kaggle)
│
├── models/                     # Model artifacts (generated after training)
│   ├── insurance_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_names.pkl
│   └── model_metadata.pkl
│
├── static/
│   ├── css/
│   │   └── style.css           # Stylesheet
│   ├── js/
│   │   └── script.js           # Frontend JavaScript
│   └── plots/                  # EDA visualizations (generated after training)
│       ├── distribution.png
│       ├── charges_by_smoker.png
│       ├── age_vs_charges.png
│       ├── bmi_categories.png
│       ├── correlation.png
│       ├── charges_by_region.png
│       └── feature_importance.png
│
└── templates/
    ├── index.html              # Prediction form page
    ├── results.html            # Prediction results page
    └── analysis.html           # EDA dashboard page
```

## 📊 Model Performance

The best model is selected based on test R² score and RMSE. Typical performance metrics:

- **R² Score**: 0.75 - 0.85 (good model performance)
- **RMSE**: $4,000 - $6,000
- **MAE**: $2,500 - $4,000
- **MAPE**: 30% - 50%

### Key Insights from Data

1. **Smoking Status**: Strongest predictor (~$20,000+ increase for smokers)
2. **Age**: Positive correlation with charges
3. **BMI**: Moderate impact (especially obese category)
4. **Children**: ~$500-$1,000 per child
5. **Region**: Minimal impact
6. **Sex**: Relatively minor impact

## 🔌 API Documentation

### POST /api/predict

Predict insurance cost via JSON API.

**Request:**
```json
{
    "age": 45,
    "sex": "male",
    "bmi": 28.5,
    "children": 2,
    "smoker": "yes",
    "region": "southeast"
}
```

**Response:**
```json
{
    "prediction": 25000.50,
    "units": "dollars"
}
```

**Error Response:**
```json
{
    "error": ["Age must be between 18 and 100"]
}
```

**Example using curl:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": "male", "bmi": 28.5, "children": 2, "smoker": "yes", "region": "southeast"}'
```

**Example using Python:**
```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "age": 45,
    "sex": "male",
    "bmi": 28.5,
    "children": 2,
    "smoker": "yes",
    "region": "southeast"
}

response = requests.post(url, json=data)
print(response.json())
```

## 🧪 Testing

Run the test suite:

```bash
python test_app.py
```

Test cases include:
- Model loading verification
- Prediction with valid inputs
- Input validation (edge cases)
- API endpoint testing
- Error handling

## 🛠 Technologies Used

### Backend
- **Python 3.8+**
- **Flask**: Web framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **joblib**: Model persistence

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling (responsive design)
- **JavaScript**: Client-side validation
- **Chart.js**: Interactive charts

## 📸 Screenshots

### Prediction Form
The user-friendly form with real-time validation and BMI calculator.

### Results Page
Detailed prediction results with cost breakdown, comparison charts, and insights.

### Analysis Dashboard
EDA visualizations and model performance metrics.

## 🔧 Configuration

All configuration is centralized in `config.py`:

- **Model Parameters**: Hyperparameters for all models
- **Paths**: Data, models, and plots directories
- **Features**: Feature definitions
- **Validation Rules**: Input validation constraints

## 🚨 Error Handling

The application handles:

- Missing dataset
- Model not trained
- Invalid user inputs
- Prediction errors
- File not found errors

## 🎯 Future Improvements

- [ ] Add SHAP values for better feature importance explanation
- [ ] Implement confidence intervals for predictions
- [ ] Add more visualization options
- [ ] Deploy to cloud (Heroku, AWS, etc.)
- [ ] Add user authentication
- [ ] Save prediction history
- [ ] Export predictions to PDF/Excel
- [ ] Add more ML models (XGBoost, Neural Networks)
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Model versioning and A/B testing

## 📝 License

This project is for educational purposes.

## 👤 Author

Machine Learning Project - Insurance Cost Prediction System

## 🙏 Acknowledgments

- Dataset: [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- scikit-learn documentation
- Flask documentation

---

**Note:** Make sure to download the dataset and train the model before running the Flask application. The model artifacts must be present in the `models/` directory for the web application to work.

