# Quick Start Guide

## 🚀 Getting Started in 5 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
1. Download `insurance.csv` from [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
2. Place it in the `data/` folder: `data/insurance.csv`

### Step 3: Train the Model
```bash
python train_model.py
```
This will:
- Load and analyze the data
- Create visualizations
- Train 5 ML models
- Save the best model to `models/` directory

### Step 4: Start the Web Application
```bash
python app.py
```

### Step 5: Open in Browser
Navigate to: `http://localhost:5000`

---

## 📋 What You'll See

### After Training (train_model.py):
- Model performance metrics for all 5 models
- Best model selection
- 7 visualization plots in `static/plots/`
- Model artifacts in `models/` directory

### Web Application Features:
- **Prediction Form**: Enter your details to get insurance cost prediction
- **Results Page**: See prediction with cost breakdown and insights
- **Analysis Dashboard**: View EDA visualizations and model metrics

---

## ⚠️ Important Notes

1. **Dataset Required**: You must download the dataset before training
2. **Train First**: Always run `train_model.py` before `app.py`
3. **Model Files**: The `models/` directory will be created automatically

---

## 🧪 Test the Application

Run tests:
```bash
python test_app.py
```

---

## 📖 Full Documentation

See `README.md` for complete documentation.

