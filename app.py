"""
Flask Web Application for Insurance Cost Prediction
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, redirect, url_for
import config

app = Flask(__name__)

# Custom Jinja2 filter for formatting numbers with commas
@app.template_filter('currency')
def currency_filter(value):
    """Format number as currency with commas and 2 decimal places"""
    try:
        return f"{float(value):,.2f}"
    except (ValueError, TypeError):
        return str(value)

# Global variables for loaded models
model = None
scaler = None
encoders = None
feature_names = None
metadata = None

def load_artifacts():
    """Load model and preprocessing artifacts"""
    global model, scaler, encoders, feature_names, metadata
    
    try:
        model_path = config.ARTIFACTS['model']
        scaler_path = config.ARTIFACTS['scaler']
        encoders_path = config.ARTIFACTS['encoders']
        feature_names_path = config.ARTIFACTS['feature_names']
        metadata_path = config.ARTIFACTS['metadata']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please run 'python train_model.py' first.")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoders_path)
        feature_names = joblib.load(feature_names_path)
        metadata = joblib.load(metadata_path)
        
        return True
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        return False

def validate_input(data):
    """Validate user input"""
    errors = []
    
    # Age validation
    try:
        age = int(data.get('age', 0))
        if not (config.VALIDATION['age']['min'] <= age <= config.VALIDATION['age']['max']):
            errors.append(f"Age must be between {config.VALIDATION['age']['min']} and {config.VALIDATION['age']['max']}")
    except (ValueError, TypeError):
        errors.append("Age must be a valid integer")
    
    # Sex validation
    if data.get('sex') not in ['male', 'female']:
        errors.append("Sex must be 'male' or 'female'")
    
    # BMI validation
    try:
        bmi = float(data.get('bmi', 0))
        if not (config.VALIDATION['bmi']['min'] <= bmi <= config.VALIDATION['bmi']['max']):
            errors.append(f"BMI must be between {config.VALIDATION['bmi']['min']} and {config.VALIDATION['bmi']['max']}")
    except (ValueError, TypeError):
        errors.append("BMI must be a valid number")
    
    # Children validation
    try:
        children = int(data.get('children', 0))
        if not (config.VALIDATION['children']['min'] <= children <= config.VALIDATION['children']['max']):
            errors.append(f"Children must be between {config.VALIDATION['children']['min']} and {config.VALIDATION['children']['max']}")
    except (ValueError, TypeError):
        errors.append("Children must be a valid integer")
    
    # Smoker validation
    if data.get('smoker') not in ['yes', 'no']:
        errors.append("Smoker must be 'yes' or 'no'")
    
    # Region validation
    valid_regions = ['northeast', 'northwest', 'southeast', 'southwest']
    if data.get('region') not in valid_regions:
        errors.append(f"Region must be one of: {', '.join(valid_regions)}")
    
    return errors

def preprocess_input(data):
    """Preprocess input data to match training format"""
    # Create a dictionary for the input
    input_dict = {}
    
    # Numerical features
    input_dict['age'] = int(data['age'])
    input_dict['bmi'] = float(data['bmi'])
    input_dict['children'] = int(data['children'])
    
    # Encode categorical features
    input_dict['sex_encoded'] = encoders['sex'].transform([data['sex']])[0]
    input_dict['smoker_encoded'] = encoders['smoker'].transform([data['smoker']])[0]
    
    # One-hot encode region
    for region_col in encoders['region_dummies']:
        base_region = region_col.replace('region_', '').lower()
        input_dict[region_col] = 1 if base_region == data['region'].lower() else 0
    
    # Engineered features
    input_dict['smoker_bmi'] = input_dict['smoker_encoded'] * input_dict['bmi']
    input_dict['smoker_age'] = input_dict['smoker_encoded'] * input_dict['age']
    input_dict['age_bmi'] = input_dict['age'] * input_dict['bmi']
    
    # Create DataFrame with feature names in correct order
    feature_vector = np.zeros(len(feature_names))
    for i, feature in enumerate(feature_names):
        if feature in input_dict:
            feature_vector[i] = input_dict[feature]
    
    return feature_vector.reshape(1, -1)

def calculate_cost_breakdown(data, prediction):
    """Calculate cost breakdown by feature"""
    # This is a simplified breakdown - in reality, you'd need SHAP values or similar
    # For now, we'll use approximate contributions based on feature importance
    
    breakdown = {
        'base': prediction * 0.2,  # Approximate base cost
        'age': 0,
        'bmi': 0,
        'smoker': 0,
        'children': 0,
        'region': 0
    }
    
    age = int(data['age'])
    bmi = float(data['bmi'])
    smoker = data['smoker'] == 'yes'
    children = int(data['children'])
    
    # Age contribution (rough estimate: ~$200 per year over 18)
    breakdown['age'] = (age - 18) * 200
    
    # BMI contribution (rough estimate: ~$150 per BMI point over 25)
    if bmi > 25:
        breakdown['bmi'] = (bmi - 25) * 150
    
    # Smoker contribution (major factor: ~$20,000)
    if smoker:
        breakdown['smoker'] = 20000
    
    # Children contribution (~$500 per child)
    breakdown['children'] = children * 500
    
    # Adjust base to make total match prediction
    total_contributions = sum(breakdown.values()) - breakdown['base']
    if total_contributions > 0:
        breakdown['base'] = max(0, prediction - total_contributions)
    
    return breakdown

@app.route('/')
def index():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if model is None:
        return render_template('index.html', error="Model not loaded. Please run 'python train_model.py' first.")
    
    try:
        # Validate input
        errors = validate_input(request.form)
        if errors:
            return render_template('index.html', errors=errors, form_data=request.form)
        
        # Preprocess input
        feature_vector = preprocess_input(request.form)
        
        # Scale features if needed
        if metadata['model_type'] in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            feature_vector_scaled = scaler.transform(feature_vector)
        else:
            feature_vector_scaled = feature_vector
        
        # Make prediction
        prediction = model.predict(feature_vector_scaled)[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Calculate breakdown
        breakdown = calculate_cost_breakdown(request.form, prediction)
        
        # Calculate average for comparison (from metadata if available, else use default)
        avg_cost = 13270  # Approximate average from dataset
        
        # Generate insights
        insights = []
        if request.form['smoker'] == 'yes':
            insights.append("🚬 Smoking significantly increases your premium")
        if float(request.form['bmi']) > 30:
            insights.append("⚠️ High BMI affects your premium")
        if float(request.form['bmi']) < 18.5:
            insights.append("⚠️ Low BMI may affect your premium")
        
        return render_template('results.html',
                             prediction=prediction,
                             breakdown=breakdown,
                             form_data=request.form,
                             avg_cost=avg_cost,
                             insights=insights,
                             metadata=metadata)
    
    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}", form_data=request.form)

@app.route('/analysis')
def analysis():
    """EDA Dashboard page"""
    plots_dir = config.PATHS['plots']
    plots = []
    
    plot_files = [
        'distribution.png',
        'charges_by_smoker.png',
        'age_vs_charges.png',
        'bmi_categories.png',
        'correlation.png',
        'charges_by_region.png',
        'feature_importance.png'
    ]
    
    for plot_file in plot_files:
        plot_path = os.path.join(plots_dir, plot_file)
        if os.path.exists(plot_path):
            plots.append(plot_file)
    
    return render_template('analysis.html', plots=plots, metadata=metadata)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for predictions"""
    if model is None:
        return jsonify({'error': "Model not loaded. Please run 'python train_model.py' first."}), 500
    
    try:
        data = request.get_json()
        
        # Validate input
        errors = validate_input(data)
        if errors:
            return jsonify({'error': errors}), 400
        
        # Preprocess input
        feature_vector = preprocess_input(data)
        
        # Scale features if needed
        if metadata['model_type'] in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            feature_vector_scaled = scaler.transform(feature_vector)
        else:
            feature_vector_scaled = feature_vector
        
        # Make prediction
        prediction = model.predict(feature_vector_scaled)[0]
        prediction = max(0, prediction)
        
        return jsonify({
            'prediction': round(prediction, 2),
            'units': 'dollars'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.before_request
def before_request():
    """Ensure artifacts are loaded before each request"""
    global model
    if model is None:
        load_artifacts()

if __name__ == '__main__':
    # Load artifacts at startup
    if load_artifacts():
        print("✅ Model and artifacts loaded successfully")
        print("🌐 Starting Flask application...")
        print("📱 Access the app at: http://localhost:5000")
    else:
        print("❌ Failed to load model. Please run 'python train_model.py' first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

