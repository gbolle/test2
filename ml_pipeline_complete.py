"""
Complete Machine Learning Pipeline for Crop Yield Prediction
=============================================================

This script performs:
1. Feature Engineering from scratch
2. Model Development and Training
3. Model Testing and Evaluation
4. Predictions and Analysis

Author: AgriSet Project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import pickle


class CropYieldMLPipeline:
    """
    Complete ML Pipeline for Crop Yield Prediction with Feature Engineering
    """
    
    def __init__(self, data_file):
        """
        Initialize the ML pipeline.
        
        Args:
            data_file (str): Path to the merged crop and rainfall dataset
        """
        self.data_file = data_file
        self.df_raw = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        
        print("="*80)
        print(" "*20 + "CROP YIELD PREDICTION ML PIPELINE")
        print(" "*25 + "Feature Engineering & Modeling")
        print("="*80)
    
    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    
    def load_data(self):
        """Load the merged crop and rainfall dataset."""
        print("\n" + "="*80)
        print("STEP 1: LOADING DATA")
        print("="*80)
        
        print(f"\nReading file: {self.data_file}")
        
        if self.data_file.endswith('.xlsx') or self.data_file.endswith('.xls'):
            self.df_raw = pd.read_excel(self.data_file)
        else:
            self.df_raw = pd.read_csv(self.data_file)
        
        print(f"✓ Loaded {len(self.df_raw):,} records")
        print(f"✓ Columns: {self.df_raw.shape[1]}")
        print(f"\nColumn names:")
        for i, col in enumerate(self.df_raw.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\nData types:")
        print(self.df_raw.dtypes)
        
        print(f"\nBasic statistics:")
        print(self.df_raw.describe())
        
        return self.df_raw
    
    # ========================================================================
    # STEP 2: DATA CLEANING
    # ========================================================================
    
    def clean_data(self):
        """Clean the dataset by handling missing values and outliers."""
        print("\n" + "="*80)
        print("STEP 2: DATA CLEANING")
        print("="*80)
        
        print(f"\nInitial records: {len(self.df_raw):,}")
        
        # Check missing values
        print("\nMissing values per column:")
        missing = self.df_raw.isnull().sum()
        missing_pct = (missing / len(self.df_raw) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Remove rows with missing target variable (Yield)
        before = len(self.df_raw)
        self.df_raw = self.df_raw.dropna(subset=['Yield'])
        after = len(self.df_raw)
        print(f"\n✓ Removed {before - after:,} rows with missing Yield")
        
        # Remove invalid yield values (zero or negative)
        before = len(self.df_raw)
        self.df_raw = self.df_raw[self.df_raw['Yield'] > 0]
        after = len(self.df_raw)
        print(f"✓ Removed {before - after:,} rows with invalid Yield (≤0)")
        
        # Remove outliers using IQR method for Yield
        Q1 = self.df_raw['Yield'].quantile(0.25)
        Q3 = self.df_raw['Yield'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        before = len(self.df_raw)
        self.df_raw = self.df_raw[
            (self.df_raw['Yield'] >= lower_bound) & 
            (self.df_raw['Yield'] <= upper_bound)
        ]
        after = len(self.df_raw)
        print(f"✓ Removed {before - after:,} outliers using IQR method (3x IQR)")
        
        print(f"\n✓ Final clean records: {len(self.df_raw):,}")
        
        return self.df_raw
    
    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    
    def feature_engineering(self):
        """Create new features from existing data."""
        print("\n" + "="*80)
        print("STEP 3: FEATURE ENGINEERING FROM SCRATCH")
        print("="*80)
        
        self.df_processed = self.df_raw.copy()
        feature_count = 0
        
        print("\n--- Creating New Features ---\n")
        
        # 3.1: Productivity Feature (Production per unit Area)
        if 'Production' in self.df_processed.columns and 'Area' in self.df_processed.columns:
            self.df_processed['Productivity'] = self.df_processed['Production'] / (self.df_processed['Area'] + 1)
            print(f"✓ Created: Productivity = Production / Area")
            feature_count += 1
        
        # 3.2: Area Categories (Small, Medium, Large farms)
        if 'Area' in self.df_processed.columns:
            area_25 = self.df_processed['Area'].quantile(0.33)
            area_75 = self.df_processed['Area'].quantile(0.67)
            
            def categorize_area(area):
                if area < area_25:
                    return 'Small'
                elif area < area_75:
                    return 'Medium'
                else:
                    return 'Large'
            
            self.df_processed['Area_Category'] = self.df_processed['Area'].apply(categorize_area)
            print(f"✓ Created: Area_Category (Small/Medium/Large)")
            feature_count += 1
        
        # 3.3: Temperature Range (difference between max and min)
        if 'Avg_Temp_Max' in self.df_processed.columns and 'Avg_Temp_Min' in self.df_processed.columns:
            self.df_processed['Temp_Range'] = self.df_processed['Avg_Temp_Max'] - self.df_processed['Avg_Temp_Min']
            print(f"✓ Created: Temp_Range = Temp_Max - Temp_Min")
            feature_count += 1
        
        # 3.4: Average Temperature
        if 'Avg_Temp_Max' in self.df_processed.columns and 'Avg_Temp_Min' in self.df_processed.columns:
            self.df_processed['Temp_Avg'] = (self.df_processed['Avg_Temp_Max'] + self.df_processed['Avg_Temp_Min']) / 2
            print(f"✓ Created: Temp_Avg = (Temp_Max + Temp_Min) / 2")
            feature_count += 1
        
        # 3.5: Humidity Range
        if 'Avg_Humidity_Max' in self.df_processed.columns and 'Avg_Humidity_Min' in self.df_processed.columns:
            self.df_processed['Humidity_Range'] = self.df_processed['Avg_Humidity_Max'] - self.df_processed['Avg_Humidity_Min']
            print(f"✓ Created: Humidity_Range = Humidity_Max - Humidity_Min")
            feature_count += 1
        
        # 3.6: Average Humidity
        if 'Avg_Humidity_Max' in self.df_processed.columns and 'Avg_Humidity_Min' in self.df_processed.columns:
            self.df_processed['Humidity_Avg'] = (self.df_processed['Avg_Humidity_Max'] + self.df_processed['Avg_Humidity_Min']) / 2
            print(f"✓ Created: Humidity_Avg = (Humidity_Max + Humidity_Min) / 2")
            feature_count += 1
        
        # 3.7: Rainfall Categories
        if 'Total_Rainfall' in self.df_processed.columns:
            rainfall_33 = self.df_processed['Total_Rainfall'].quantile(0.33)
            rainfall_67 = self.df_processed['Total_Rainfall'].quantile(0.67)
            
            def categorize_rainfall(rainfall):
                if pd.isna(rainfall):
                    return 'Unknown'
                elif rainfall < rainfall_33:
                    return 'Low'
                elif rainfall < rainfall_67:
                    return 'Medium'
                else:
                    return 'High'
            
            self.df_processed['Rainfall_Category'] = self.df_processed['Total_Rainfall'].apply(categorize_rainfall)
            print(f"✓ Created: Rainfall_Category (Low/Medium/High)")
            feature_count += 1
        
        # 3.8: Rainfall per Day (Total Rainfall / Season Days)
        # Kharif: Jun-Oct (153 days), Rabi: Nov-Mar (121 days)
        if 'Total_Rainfall' in self.df_processed.columns and 'Season' in self.df_processed.columns:
            season_days = {'Kharif': 153, 'Rabi': 121}
            self.df_processed['Rainfall_Per_Day'] = self.df_processed.apply(
                lambda row: row['Total_Rainfall'] / season_days.get(row['Season'], 137) if pd.notna(row['Total_Rainfall']) else 0,
                axis=1
            )
            print(f"✓ Created: Rainfall_Per_Day = Total_Rainfall / Season_Days")
            feature_count += 1
        
        # 3.9: Growing Degree Days (GDD) - simplified calculation
        if 'Temp_Avg' in self.df_processed.columns:
            base_temp = 10  # Base temperature for crop growth
            self.df_processed['GDD'] = self.df_processed['Temp_Avg'].apply(
                lambda x: max(0, x - base_temp) if pd.notna(x) else 0
            )
            print(f"✓ Created: GDD (Growing Degree Days) = max(0, Temp_Avg - 10)")
            feature_count += 1
        
        # 3.10: Heat Stress Indicator (days with extreme temperature)
        if 'Avg_Temp_Max' in self.df_processed.columns:
            self.df_processed['Heat_Stress'] = (self.df_processed['Avg_Temp_Max'] > 35).astype(int)
            print(f"✓ Created: Heat_Stress (1 if Temp_Max > 35°C, else 0)")
            feature_count += 1
        
        # 3.11: Water Stress Indicator (low rainfall + high temperature)
        if 'Total_Rainfall' in self.df_processed.columns and 'Temp_Avg' in self.df_processed.columns:
            rainfall_median = self.df_processed['Total_Rainfall'].median()
            temp_median = self.df_processed['Temp_Avg'].median()
            
            self.df_processed['Water_Stress'] = (
                (self.df_processed['Total_Rainfall'] < rainfall_median) & 
                (self.df_processed['Temp_Avg'] > temp_median)
            ).astype(int)
            print(f"✓ Created: Water_Stress (low rainfall + high temp)")
            feature_count += 1
        
        # 3.12: Optimal Conditions Indicator
        if 'Total_Rainfall' in self.df_processed.columns and 'Temp_Avg' in self.df_processed.columns:
            rainfall_optimal_low = self.df_processed['Total_Rainfall'].quantile(0.4)
            rainfall_optimal_high = self.df_processed['Total_Rainfall'].quantile(0.8)
            temp_optimal_low = 20
            temp_optimal_high = 30
            
            self.df_processed['Optimal_Conditions'] = (
                (self.df_processed['Total_Rainfall'].between(rainfall_optimal_low, rainfall_optimal_high)) &
                (self.df_processed['Temp_Avg'].between(temp_optimal_low, temp_optimal_high))
            ).astype(int)
            print(f"✓ Created: Optimal_Conditions (ideal rainfall + temperature)")
            feature_count += 1
        
        # 3.13: Year-based trend feature
        if 'Year' in self.df_processed.columns:
            min_year = self.df_processed['Year'].min()
            self.df_processed['Years_Since_Start'] = self.df_processed['Year'] - min_year
            print(f"✓ Created: Years_Since_Start (years since {min_year})")
            feature_count += 1
        
        # 3.14: Interaction features
        if 'Area' in self.df_processed.columns and 'Total_Rainfall' in self.df_processed.columns:
            self.df_processed['Area_Rainfall_Interaction'] = self.df_processed['Area'] * self.df_processed['Total_Rainfall'].fillna(0)
            print(f"✓ Created: Area_Rainfall_Interaction = Area × Rainfall")
            feature_count += 1
        
        if 'Area' in self.df_processed.columns and 'Temp_Avg' in self.df_processed.columns:
            self.df_processed['Area_Temp_Interaction'] = self.df_processed['Area'] * self.df_processed['Temp_Avg'].fillna(0)
            print(f"✓ Created: Area_Temp_Interaction = Area × Temperature")
            feature_count += 1
        
        print(f"\n{'='*80}")
        print(f"✓ Total new features created: {feature_count}")
        print(f"{'='*80}")
        
        return self.df_processed
    
    # ========================================================================
    # STEP 4: FEATURE SELECTION AND ENCODING
    # ========================================================================
    
    def prepare_features(self):
        """Select features and encode categorical variables."""
        print("\n" + "="*80)
        print("STEP 4: FEATURE SELECTION & ENCODING")
        print("="*80)
        
        # Encode categorical variables
        print("\n--- Encoding Categorical Variables ---\n")
        
        categorical_features = ['District', 'Season', 'Crop', 'Area_Category', 'Rainfall_Category']
        
        for cat_col in categorical_features:
            if cat_col in self.df_processed.columns:
                self.label_encoders[cat_col] = LabelEncoder()
                self.df_processed[f'{cat_col}_Encoded'] = self.label_encoders[cat_col].fit_transform(
                    self.df_processed[cat_col].astype(str)
                )
                n_unique = len(self.label_encoders[cat_col].classes_)
                print(f"✓ Encoded {cat_col}: {n_unique} unique values")
        
        # Select features for modeling
        print("\n--- Selecting Features for Modeling ---\n")
        
        # Numerical features
        numerical_features = [
            'Year', 'Area', 'Years_Since_Start',
            'Total_Rainfall', 'Rainfall_Per_Day',
            'Avg_Temp_Min', 'Avg_Temp_Max', 'Temp_Avg', 'Temp_Range',
            'Avg_Humidity_Min', 'Avg_Humidity_Max', 'Humidity_Avg', 'Humidity_Range',
            'GDD', 'Heat_Stress', 'Water_Stress', 'Optimal_Conditions',
            'Area_Rainfall_Interaction', 'Area_Temp_Interaction', 'Productivity'
        ]
        
        # Encoded categorical features
        encoded_features = [f'{cat}_Encoded' for cat in categorical_features if cat in self.df_processed.columns]
        
        # Combine all features
        self.feature_names = [f for f in numerical_features + encoded_features if f in self.df_processed.columns]
        
        print(f"Selected {len(self.feature_names)} features:")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"  {i:2d}. {feat}")
        
        # Prepare X and y
        X = self.df_processed[self.feature_names].copy()
        y = self.df_processed['Yield'].copy()
        
        # Handle any remaining missing values - CRITICAL FIX
        print(f"\n--- Handling Missing Values ---\n")
        
        # Show missing values before
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Missing values found in {(missing_counts > 0).sum()} columns:")
            for col in missing_counts[missing_counts > 0].index:
                print(f"  {col}: {missing_counts[col]} missing")
        
        # Fill missing values with median for numerical columns
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                if pd.isna(median_val):  # If median is also NaN, use 0
                    X[col] = X[col].fillna(0)
                    print(f"  Filled {col} NaN with 0 (median was NaN)")
                else:
                    X[col] = X[col].fillna(median_val)
        
        # Final check - replace any remaining NaN with 0
        X = X.fillna(0)
        
        # Verify no NaN values remain
        remaining_nan = X.isnull().sum().sum()
        if remaining_nan > 0:
            print(f"⚠ Warning: {remaining_nan} NaN values still present - replacing with 0")
            X = X.fillna(0)
        else:
            print(f"✓ All missing values handled successfully")
        
        # Also check for infinite values
        if np.isinf(X.values).any():
            print(f"⚠ Warning: Infinite values found - replacing with 0")
            X = X.replace([np.inf, -np.inf], 0)
        
        print(f"\n✓ Feature matrix shape: {X.shape}")
        print(f"✓ Target variable shape: {y.shape}")
        print(f"✓ No NaN values: {X.isnull().sum().sum() == 0}")
        print(f"✓ No Inf values: {not np.isinf(X.values).any()}")
        print(f"\nTarget (Yield) statistics:")
        print(f"  Min:    {y.min():.2f}")
        print(f"  Max:    {y.max():.2f}")
        print(f"  Mean:   {y.mean():.2f}")
        print(f"  Median: {y.median():.2f}")
        print(f"  Std:    {y.std():.2f}")
        
        return X, y
    
    # ========================================================================
    # STEP 5: TRAIN-TEST SPLIT
    # ========================================================================
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print("\n" + "="*80)
        print("STEP 5: TRAIN-TEST SPLIT")
        print("="*80)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nSplit ratio: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        print(f"✓ Training set:   {len(self.X_train):,} samples")
        print(f"✓ Testing set:    {len(self.X_test):,} samples")
        
        print(f"\nTraining target statistics:")
        print(f"  Mean:   {self.y_train.mean():.2f}")
        print(f"  Median: {self.y_train.median():.2f}")
        print(f"  Std:    {self.y_train.std():.2f}")
        
        print(f"\nTesting target statistics:")
        print(f"  Mean:   {self.y_test.mean():.2f}")
        print(f"  Median: {self.y_test.median():.2f}")
        print(f"  Std:    {self.y_test.std():.2f}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # ========================================================================
    # STEP 6: FEATURE SCALING
    # ========================================================================
    
    def scale_features(self):
        """Scale features using RobustScaler."""
        print("\n" + "="*80)
        print("STEP 6: FEATURE SCALING")
        print("="*80)
        
        print("\nUsing RobustScaler (robust to outliers)")
        
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ Scaled training set: {self.X_train_scaled.shape}")
        print(f"✓ Scaled testing set:  {self.X_test_scaled.shape}")
        
        # Convert back to DataFrame for easier handling
        self.X_train_scaled = pd.DataFrame(
            self.X_train_scaled, 
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.X_test_scaled, 
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        return self.X_train_scaled, self.X_test_scaled
    
    # ========================================================================
    # STEP 7: MODEL TRAINING
    # ========================================================================
    
    def train_models(self):
        """Train multiple regression models."""
        print("\n" + "="*80)
        print("STEP 7: MODEL TRAINING & EVALUATION")
        print("="*80)
        
        # Define models to train
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            ),
            'Extra Trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1, max_iter=10000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        }
        
        results = {}
        
        print(f"\nTraining {len(self.models)} models...\n")
        
        for model_name, model in self.models.items():
            print(f"{'─'*80}")
            print(f"Training: {model_name}")
            print(f"{'─'*80}")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_train_pred = model.predict(self.X_train_scaled)
            y_test_pred = model.predict(self.X_test_scaled)
            
            # Calculate metrics
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            train_r2 = r2_score(self.y_train, y_train_pred)
            
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='r2', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[model_name] = {
                'model': model,
                'train_mae': train_mae,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'y_pred': y_test_pred
            }
            
            print(f"\nTraining Metrics:")
            print(f"  MAE:  {train_mae:.4f}")
            print(f"  RMSE: {train_rmse:.4f}")
            print(f"  R²:   {train_r2:.4f}")
            
            print(f"\nTesting Metrics:")
            print(f"  MAE:  {test_mae:.4f}")
            print(f"  RMSE: {test_rmse:.4f}")
            print(f"  R²:   {test_r2:.4f}")
            
            print(f"\nCross-Validation (5-fold):")
            print(f"  Mean R²: {cv_mean:.4f} (+/- {cv_std:.4f})")
            print()
        
        # Select best model based on test R²
        self.best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"{'='*80}")
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"{'='*80}")
        print(f"\nTest Set Performance:")
        print(f"  MAE:  {results[self.best_model_name]['test_mae']:.4f}")
        print(f"  RMSE: {results[self.best_model_name]['test_rmse']:.4f}")
        print(f"  R²:   {results[self.best_model_name]['test_r2']:.4f}")
        print(f"\nCross-Validation:")
        print(f"  Mean R²: {results[self.best_model_name]['cv_mean']:.4f} (+/- {results[self.best_model_name]['cv_std']:.4f})")
        
        return results
    
    # ========================================================================
    # STEP 8: FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    
    def analyze_feature_importance(self):
        """Analyze and display feature importance."""
        print("\n" + "="*80)
        print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print(f"\nTop 15 Most Important Features:\n")
            for i, row in feature_importance_df.head(15).iterrows():
                bar = '█' * int(row['Importance'] * 100)
                print(f"  {row['Feature']:<30} {row['Importance']:.4f} {bar}")
            
            return feature_importance_df
        else:
            print(f"\n{self.best_model_name} does not provide feature importances.")
            
            # For linear models, show coefficients
            if hasattr(self.best_model, 'coef_'):
                coef_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Coefficient': self.best_model.coef_
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                print(f"\nTop 15 Features by Coefficient Magnitude:\n")
                for i, row in coef_df.head(15).iterrows():
                    print(f"  {row['Feature']:<30} {row['Coefficient']:>10.4f}")
                
                return coef_df
            
            return None
    
    # ========================================================================
    # STEP 9: SAVE RESULTS
    # ========================================================================
    
    def save_results(self, results, output_file, model_file):
        """Save predictions and model to files."""
        print("\n" + "="*80)
        print("STEP 9: SAVING RESULTS")
        print("="*80)
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'Actual_Yield': self.y_test.values,
            'Predicted_Yield': results[self.best_model_name]['y_pred'],
            'Absolute_Error': abs(self.y_test.values - results[self.best_model_name]['y_pred']),
            'Percentage_Error': abs(self.y_test.values - results[self.best_model_name]['y_pred']) / self.y_test.values * 100
        })
        
        # Add all model predictions
        for model_name in results:
            predictions_df[f'{model_name}_Predicted'] = results[model_name]['y_pred']
        
        # Save to Excel
        print(f"\nSaving predictions to: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Predictions sheet
            predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
            
            # Model comparison sheet
            comparison_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Test_MAE': [results[m]['test_mae'] for m in results],
                'Test_RMSE': [results[m]['test_rmse'] for m in results],
                'Test_R2': [results[m]['test_r2'] for m in results],
                'CV_Mean_R2': [results[m]['cv_mean'] for m in results],
                'CV_Std': [results[m]['cv_std'] for m in results]
            }).sort_values('Test_R2', ascending=False)
            comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
            
            # Feature importance (if available)
            importance = self.analyze_feature_importance()
            if importance is not None:
                importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
        
        print(f"✓ Predictions saved successfully")
        
        # Save model
        print(f"\nSaving trained model to: {model_file}")
        model_package = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'metrics': {
                'test_mae': results[self.best_model_name]['test_mae'],
                'test_rmse': results[self.best_model_name]['test_rmse'],
                'test_r2': results[self.best_model_name]['test_r2']
            }
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✓ Model saved successfully")
        
        return predictions_df
    
    # ========================================================================
    # MAIN PIPELINE EXECUTION
    # ========================================================================
    
    def run_pipeline(self, output_file, model_file):
        """Run the complete ML pipeline."""
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Feature engineering
        self.feature_engineering()
        
        # Step 4: Prepare features
        X, y = self.prepare_features()
        
        # Step 5: Split data
        self.split_data(X, y)
        
        # Step 6: Scale features
        self.scale_features()
        
        # Step 7: Train models
        results = self.train_models()
        
        # Step 8: Feature importance
        self.analyze_feature_importance()
        
        # Step 9: Save results
        self.save_results(results, output_file, model_file)
        
        print("\n" + "="*80)
        print("✓ PIPELINE EXECUTION COMPLETE!")
        print("="*80)
        print(f"\nOutput files created:")
        print(f"  ✓ Predictions: {output_file}")
        print(f"  ✓ Model:       {model_file}")
        print(f"\n🌾 Your crop yield prediction model is ready to use!")
        print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the ML pipeline."""
    
    # UPDATE THESE PATHS FOR YOUR SYSTEM
    data_file = r"D:\AgriSet Project\Data\output\Telangana_Crop_Rainfall_Merged.xlsx"
    output_file = r"D:\AgriSet Project\Data\output\ML_Predictions_Results.xlsx"
    model_file = r"D:\AgriSet Project\Data\output\crop_yield_model.pkl"
    
    # Run the pipeline
    pipeline = CropYieldMLPipeline(data_file)
    pipeline.run_pipeline(output_file, model_file)


if __name__ == "__main__":
    main()
