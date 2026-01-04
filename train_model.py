"""
Insurance Cost Prediction - Model Training Script
Performs EDA, preprocessing, model training, and evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import config

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def create_bmi_category(bmi):
    """Create BMI category based on WHO standards"""
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def create_age_group(age):
    """Create age group"""
    if age < 30:
        return 'Young Adult'
    elif 30 <= age < 50:
        return 'Middle Aged'
    else:
        return 'Senior'

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def load_and_inspect_data():
    """Load and inspect the dataset"""
    print("=" * 60)
    print("Phase 1: Data Loading and Inspection")
    print("=" * 60)
    
    data_path = config.PATHS['data']
    
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Dataset not found at {data_path}")
        print("Please download the insurance dataset and place it in the data/ directory")
        print("Dataset URL: https://www.kaggle.com/datasets/mirichoi0218/insurance")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n📋 Column Information:")
    print(df.info())
    
    print("\n📈 Statistical Summary:")
    print(df.describe())
    
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values found!")
    else:
        print(missing[missing > 0])
    
    print("\n🔍 Checking for Outliers (IQR Method):")
    numerical_cols = config.FEATURES['numerical'] + [config.FEATURES['target']]
    for col in numerical_cols:
        outliers, lower, upper = detect_outliers_iqr(df, col)
        print(f"   {col}: {len(outliers)} outliers (Range: [{lower:.2f}, {upper:.2f}])")
    
    print("\n📊 First few rows:")
    print(df.head())
    
    return df

def perform_eda(df):
    """Perform Exploratory Data Analysis and save plots"""
    print("\n" + "=" * 60)
    print("Phase 1.2: Exploratory Data Analysis")
    print("=" * 60)
    
    plots_dir = config.PATHS['plots']
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Distribution of charges (histogram with KDE)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], kde=True, bins=50)
    plt.title('Distribution of Insurance Charges', fontsize=16, fontweight='bold')
    plt.xlabel('Charges ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: distribution.png")
    
    # 2. Charges by smoker status (box plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='smoker', y='charges')
    plt.title('Insurance Charges by Smoker Status', fontsize=16, fontweight='bold')
    plt.xlabel('Smoker', fontsize=12)
    plt.ylabel('Charges ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'charges_by_smoker.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: charges_by_smoker.png")
    
    # 3. Age vs charges scatter plot (color by smoker)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker', alpha=0.6)
    plt.title('Age vs Insurance Charges (colored by Smoker Status)', fontsize=16, fontweight='bold')
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Charges ($)', fontsize=12)
    plt.legend(title='Smoker')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'age_vs_charges.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: age_vs_charges.png")
    
    # 4. BMI categories vs charges
    df['bmi_category'] = df['bmi'].apply(create_bmi_category)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='bmi_category', y='charges', order=['Underweight', 'Normal', 'Overweight', 'Obese'])
    plt.title('Insurance Charges by BMI Category', fontsize=16, fontweight='bold')
    plt.xlabel('BMI Category', fontsize=12)
    plt.ylabel('Charges ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bmi_categories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: bmi_categories.png")
    
    # 5. Correlation heatmap
    plt.figure(figsize=(10, 8))
    numerical_df = df[config.FEATURES['numerical'] + [config.FEATURES['target']]]
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: correlation.png")
    
    # 6. Charges distribution by region
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='region', y='charges')
    plt.title('Insurance Charges by Region', fontsize=16, fontweight='bold')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Charges ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'charges_by_region.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: charges_by_region.png")
    
    # Drop temporary column
    df = df.drop('bmi_category', axis=1)

def preprocess_data(df):
    """Preprocess and engineer features"""
    print("\n" + "=" * 60)
    print("Phase 1.3-1.5: Feature Engineering and Preprocessing")
    print("=" * 60)
    
    df_processed = df.copy()
    
    # 1.4 Encoding Categorical Variables
    print("\n📝 Encoding categorical variables...")
    
    # Label encoding for sex (male=1, female=0)
    le_sex = LabelEncoder()
    df_processed['sex_encoded'] = le_sex.fit_transform(df_processed['sex'])
    
    # Label encoding for smoker (yes=1, no=0)
    le_smoker = LabelEncoder()
    df_processed['smoker_encoded'] = le_smoker.fit_transform(df_processed['smoker'])
    
    # One-hot encoding for region (drop first to avoid multicollinearity)
    region_dummies = pd.get_dummies(df_processed['region'], prefix='region', drop_first=True)
    df_processed = pd.concat([df_processed, region_dummies], axis=1)
    
    # Store encoders
    encoders = {
        'sex': le_sex,
        'smoker': le_smoker,
        'region_dummies': list(region_dummies.columns)
    }
    
    # 1.3 Feature Engineering
    print("🔧 Creating engineered features...")
    
    # Interaction features
    df_processed['smoker_bmi'] = df_processed['smoker_encoded'] * df_processed['bmi']
    df_processed['smoker_age'] = df_processed['smoker_encoded'] * df_processed['age']
    df_processed['age_bmi'] = df_processed['age'] * df_processed['bmi']
    
    # Prepare feature columns
    feature_cols = (config.FEATURES['numerical'] + 
                   ['sex_encoded', 'smoker_encoded'] + 
                   list(region_dummies.columns) + 
                   config.FEATURES['engineered'])
    
    X = df_processed[feature_cols]
    y = df_processed[config.FEATURES['target']]
    
    print(f"✅ Final feature set: {len(feature_cols)} features")
    print(f"   Features: {', '.join(feature_cols)}")
    
    return X, y, encoders, feature_cols

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models"""
    print("\n" + "=" * 60)
    print("Phase 2: Model Training and Evaluation")
    print("=" * 60)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.MODEL_CONFIG['test_size'],
        random_state=config.MODEL_CONFIG['random_state']
    )
    
    print(f"\n📊 Train-Test Split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Feature scaling
    print("\n📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=config.MODEL_CONFIG['models']['ridge']['alpha']),
        'Lasso Regression': Lasso(alpha=config.MODEL_CONFIG['models']['lasso']['alpha']),
        'Random Forest': RandomForestRegressor(
            n_estimators=config.MODEL_CONFIG['models']['random_forest']['n_estimators'],
            max_depth=config.MODEL_CONFIG['models']['random_forest']['max_depth'],
            min_samples_split=config.MODEL_CONFIG['models']['random_forest']['min_samples_split'],
            random_state=config.MODEL_CONFIG['models']['random_forest']['random_state']
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=config.MODEL_CONFIG['models']['gradient_boosting']['n_estimators'],
            learning_rate=config.MODEL_CONFIG['models']['gradient_boosting']['learning_rate'],
            max_depth=config.MODEL_CONFIG['models']['gradient_boosting']['max_depth'],
            random_state=config.MODEL_CONFIG['models']['gradient_boosting']['random_state']
        )
    }
    
    results = {}
    cv_folds = config.MODEL_CONFIG['cv_folds']
    
    print("\n" + "=" * 60)
    print("Training and Evaluating Models")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n🔬 Training {name}...")
        
        # Determine if scaling is needed
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            X_tr = X_train_scaled
            X_te = X_test_scaled
        else:
            X_tr = X_train.values
            X_te = X_test.values
        
        # Train model
        model.fit(X_tr, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_tr)
        y_test_pred = model.predict(X_te)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_mape = calculate_mape(y_train, y_train_pred)
        test_mape = calculate_mape(y_test, y_test_pred)
        
        # Cross-validation
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            cv_scores = cross_val_score(model, X_tr, y_train, cv=cv_folds, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_tr, y_train, cv=cv_folds, scoring='r2')
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'uses_scaling': name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
        }
        
        # Print results
        print(f"   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        print(f"   Train RMSE: ${train_rmse:.2f} | Test RMSE: ${test_rmse:.2f}")
        print(f"   Train MAE: ${train_mae:.2f} | Test MAE: ${test_mae:.2f}")
        print(f"   Train MAPE: {train_mape:.2f}% | Test MAPE: {test_mape:.2f}%")
        print(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results, scaler, X_train, X_test, y_train, y_test

def select_best_model(results):
    """Select the best model based on test R² and RMSE"""
    print("\n" + "=" * 60)
    print("Phase 2.5: Model Selection")
    print("=" * 60)
    
    # Find best model (highest test R², lowest test RMSE)
    best_model_name = max(results.keys(), key=lambda x: (results[x]['test_r2'], -results[x]['test_rmse']))
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"   Test RMSE: ${results[best_model_name]['test_rmse']:.2f}")
    print(f"   Test MAE: ${results[best_model_name]['test_mae']:.2f}")
    print(f"   Test MAPE: {results[best_model_name]['test_mape']:.2f}%")
    
    # Check for overfitting
    train_r2 = results[best_model_name]['train_r2']
    test_r2 = results[best_model_name]['test_r2']
    if train_r2 - test_r2 > 0.1:
        print(f"\n⚠️  Warning: Potential overfitting detected (Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f})")
    else:
        print(f"\n✅ Good generalization (Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f})")
    
    return best_model_name, best_model, results[best_model_name]

def plot_feature_importance(best_model_name, best_model, feature_names):
    """Plot feature importance for tree-based models"""
    print("\n" + "=" * 60)
    print("Phase 2.6: Feature Importance")
    print("=" * 60)
    
    plots_dir = config.PATHS['plots']
    
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importance - {best_model_name}', fontsize=16, fontweight='bold')
        plt.barh(range(len(importances)), importances[indices])
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved: feature_importance.png")
        
        print("\n📊 Top 5 Most Important Features:")
        for i in range(min(5, len(indices))):
            idx = indices[i]
            print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    else:
        # For linear models, show coefficients
        if hasattr(best_model, 'coef_'):
            coef = best_model.coef_
            indices = np.argsort(np.abs(coef))[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.title(f'Feature Coefficients - {best_model_name}', fontsize=16, fontweight='bold')
            plt.barh(range(len(coef)), coef[indices])
            plt.yticks(range(len(coef)), [feature_names[i] for i in indices])
            plt.xlabel('Coefficient Value', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Saved: feature_importance.png")
            
            print("\n📊 Top 5 Most Influential Features:")
            for i in range(min(5, len(indices))):
                idx = indices[i]
                print(f"   {i+1}. {feature_names[idx]}: {coef[idx]:.4f}")

def save_artifacts(best_model, scaler, encoders, feature_names, best_model_name, metrics):
    """Save model and preprocessing artifacts"""
    print("\n" + "=" * 60)
    print("Phase 3: Model Persistence")
    print("=" * 60)
    
    models_dir = config.PATHS['models']
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = config.ARTIFACTS['model']
    joblib.dump(best_model, model_path)
    print(f"✅ Saved model: {model_path}")
    
    # Save scaler
    scaler_path = config.ARTIFACTS['scaler']
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved scaler: {scaler_path}")
    
    # Save encoders
    encoders_path = config.ARTIFACTS['encoders']
    joblib.dump(encoders, encoders_path)
    print(f"✅ Saved encoders: {encoders_path}")
    
    # Save feature names
    feature_names_path = config.ARTIFACTS['feature_names']
    joblib.dump(feature_names, feature_names_path)
    print(f"✅ Saved feature names: {feature_names_path}")
    
    # Save metadata
    metadata = {
        'model_type': best_model_name,
        'r2_score': metrics['test_r2'],
        'rmse': metrics['test_rmse'],
        'mae': metrics['test_mae'],
        'mape': metrics['test_mape'],
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'features': list(feature_names)
    }
    metadata_path = config.ARTIFACTS['metadata']
    joblib.dump(metadata, metadata_path)
    print(f"✅ Saved metadata: {metadata_path}")

def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("🚀 Insurance Cost Prediction - Model Training")
    print("=" * 60)
    
    # Phase 1: Data Exploration & Preprocessing
    df = load_and_inspect_data()
    perform_eda(df)
    X, y, encoders, feature_names = preprocess_data(df)
    
    # Phase 2: Model Training
    results, scaler, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)
    best_model_name, best_model, metrics = select_best_model(results)
    plot_feature_importance(best_model_name, best_model, feature_names)
    
    # Phase 3: Model Persistence
    save_artifacts(best_model, scaler, encoders, feature_names, best_model_name, metrics)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"\n📁 Models saved in: {config.PATHS['models']}")
    print(f"📊 Plots saved in: {config.PATHS['plots']}")
    print(f"\n🎯 Next step: Run 'python app.py' to start the Flask web application")

if __name__ == "__main__":
    main()

