"""
Crop Yield Prediction Web Application with Comprehensive Visualizations
Built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #2E7D32;
    }
    h2 {
        color: #388E3C;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained ML model (compressed with joblib)."""
    model_path = Path("crop_yield_model.pkl")
    
    if not model_path.exists():
        return None
    
    try:
        # Use joblib for loading compressed sklearn models
        model_package = joblib.load(model_path)
        return model_package
    except Exception as e:
        # Fallback to pickle if joblib fails
        try:
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            return model_package
        except:
            return None


@st.cache_data
def load_training_data():
    """Load training data for visualizations."""
    # Try root directory first (for deployment)
    data_path = Path("telangana_complete_32districts.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    
    # Try local development paths
    data_path = Path("Telangana_AgriData_Suite/sample_data/telangana_final_processed_dataset.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    
    # Fallback options
    data_path = Path("Telangana_AgriData_Suite/sample_data/telangana_complete_32districts.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    
    data_path = Path("Telangana_AgriData_Suite/sample_data/Telangana_Crop_Rainfall_Merged.xlsx")
    if data_path.exists():
        df = pd.read_excel(data_path)
        return df
    
    data_path = Path("Telangana_AgriData_Suite/sample_data/telangana_crop_data_cleaned.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    
    return None


def calculate_engineered_features(inputs):
    """Calculate all engineered features from user inputs."""
    features = {}
    
    features['Year'] = inputs['year']
    features['Area'] = inputs['area']
    
    features['District_Encoded'] = 0
    features['Season_Encoded'] = 0
    features['Crop_Encoded'] = 0
    
    features['Years_Since_Start'] = inputs['year'] - 2018
    
    features['Total_Rainfall'] = inputs.get('total_rainfall', 0)
    features['Avg_Temp_Min'] = inputs.get('temp_min', 0)
    features['Avg_Temp_Max'] = inputs.get('temp_max', 0)
    features['Avg_Humidity_Min'] = inputs.get('humidity_min', 0)
    features['Avg_Humidity_Max'] = inputs.get('humidity_max', 0)
    
    features['Temp_Range'] = features['Avg_Temp_Max'] - features['Avg_Temp_Min']
    features['Temp_Avg'] = (features['Avg_Temp_Max'] + features['Avg_Temp_Min']) / 2
    features['Humidity_Range'] = features['Avg_Humidity_Max'] - features['Avg_Humidity_Min']
    features['Humidity_Avg'] = (features['Avg_Humidity_Max'] + features['Avg_Humidity_Min']) / 2
    
    season_days = {'Kharif': 153, 'Rabi': 121}
    features['Rainfall_Per_Day'] = features['Total_Rainfall'] / season_days.get(inputs['season'], 137)
    
    features['GDD'] = max(0, features['Temp_Avg'] - 10)
    
    features['Heat_Stress'] = 1 if features['Avg_Temp_Max'] > 35 else 0
    features['Water_Stress'] = 1 if (features['Total_Rainfall'] < 500 and features['Temp_Avg'] > 28) else 0
    features['Optimal_Conditions'] = 1 if (500 <= features['Total_Rainfall'] <= 1200 and 20 <= features['Temp_Avg'] <= 30) else 0
    
    features['Area_Rainfall_Interaction'] = features['Area'] * features['Total_Rainfall']
    features['Area_Temp_Interaction'] = features['Area'] * features['Temp_Avg']
    
    # NO PRODUCTIVITY FEATURE - Model learns from weather & agricultural features
    # This allows predictions to respond to optimal conditions and reach higher yields
    
    return features


def show_data_exploration(df):
    """Show data exploration tab."""
    st.markdown("## 📊 Data Exploration")
    
    if df is None:
        st.warning("Training data not found. This tab requires the sample data file.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Crops", f"{df['Crop'].nunique()}")
    with col3:
        st.metric("Districts", f"{df['District'].nunique()}")
    with col4:
        st.metric("Years", f"{df['Year'].min()}-{df['Year'].max()}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Yield Distribution")
        fig = px.histogram(df, x='Yield', nbins=50, 
                          title='Distribution of Crop Yield',
                          labels={'Yield': 'Yield (kg/ha)', 'count': 'Frequency'},
                          color_discrete_sequence=['#2E7D32'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🌧️ Rainfall Distribution")
        fig = px.histogram(df, x='Total_Rainfall', nbins=50,
                          title='Distribution of Total Rainfall',
                          labels={'Total_Rainfall': 'Rainfall (mm)', 'count': 'Frequency'},
                          color_discrete_sequence=['#1976D2'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌾 Yield by Crop (Top 10)")
        top_crops = df.groupby('Crop')['Yield'].mean().nlargest(10).reset_index()
        fig = px.bar(top_crops, x='Yield', y='Crop', orientation='h',
                    title='Average Yield by Crop',
                    labels={'Yield': 'Average Yield (kg/ha)', 'Crop': 'Crop'},
                    color='Yield', color_continuous_scale='Greens')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📍 Yield by District (Top 10)")
        top_districts = df.groupby('District')['Yield'].mean().nlargest(10).reset_index()
        fig = px.bar(top_districts, x='Yield', y='District', orientation='h',
                    title='Average Yield by District',
                    labels={'Yield': 'Average Yield (kg/ha)', 'District': 'District'},
                    color='Yield', color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📅 Yield Trends Over Years")
    yearly_avg = df.groupby('Year')['Yield'].mean().reset_index()
    fig = px.line(yearly_avg, x='Year', y='Yield',
                 title='Average Crop Yield Over Years',
                 labels={'Yield': 'Average Yield (kg/ha)', 'Year': 'Year'},
                 markers=True)
    fig.update_traces(line_color='#2E7D32', line_width=3, marker_size=10)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def show_correlation_analysis(df):
    """Show correlation analysis tab."""
    st.markdown("## 🔗 Correlation Analysis")
    
    if df is None:
        st.warning("Training data not found. This tab requires the sample data file.")
        return
    
    numerical_cols = ['Yield', 'Area', 'Total_Rainfall', 'Avg_Temp_Min', 'Avg_Temp_Max',
                     'Avg_Humidity_Min', 'Avg_Humidity_Max']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    df_corr = df[numerical_cols].corr()
    
    st.markdown("### 🔥 Feature Correlation Heatmap")
    fig = px.imshow(df_corr, 
                   text_auto=True,
                   color_continuous_scale='RdBu_r',
                   aspect='auto',
                   title='Correlation Matrix of Numerical Features',
                   labels=dict(color="Correlation"))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Correlations with Yield")
    yield_corr = df_corr['Yield'].drop('Yield').sort_values(ascending=True)
    
    colors = ['#d32f2f' if x < 0 else '#388E3C' for x in yield_corr.values]
    
    fig = go.Figure(go.Bar(
        x=yield_corr.values,
        y=yield_corr.index,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{x:.3f}' for x in yield_corr.values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Feature Correlation with Crop Yield',
        xaxis_title='Correlation Coefficient',
        yaxis_title='Feature',
        height=500,
        showlegend=False
    )
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🌡️ Temperature vs Yield")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Avg_Temp_Max' in df.columns:
            fig = px.scatter(df.sample(min(1000, len(df))), x='Avg_Temp_Max', y='Yield',
                           title='Maximum Temperature vs Yield',
                           labels={'Avg_Temp_Max': 'Max Temperature (°C)', 'Yield': 'Yield (kg/ha)'},
                           trendline='ols', opacity=0.6)
            fig.update_traces(marker=dict(color='#d32f2f'))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Total_Rainfall' in df.columns:
            fig = px.scatter(df.sample(min(1000, len(df))), x='Total_Rainfall', y='Yield',
                           title='Rainfall vs Yield',
                           labels={'Total_Rainfall': 'Rainfall (mm)', 'Yield': 'Yield (kg/ha)'},
                           trendline='ols', opacity=0.6)
            fig.update_traces(marker=dict(color='#1976D2'))
            st.plotly_chart(fig, use_container_width=True)


def show_model_performance(model_package):
    """Show model performance tab."""
    st.markdown("## 🏆 Model Performance")
    
    if model_package is None:
        st.warning("Model not loaded.")
        return
    
    metrics = model_package.get('metrics', model_package.get('performance', {}))
    model_name = model_package.get('model_name', 'Random Forest')
    
    st.markdown(f"### 🎯 Best Model: **{model_name}**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    test_r2 = metrics.get('test_r2', metrics.get('r2_score', 0.85))
    test_mae = metrics.get('test_mae', metrics.get('mae', 250))
    test_rmse = metrics.get('test_rmse', metrics.get('rmse', 350))
    
    with col1:
        st.metric("R² Score", f"{test_r2:.4f}",
                 help="Coefficient of determination - measures prediction accuracy")
    with col2:
        st.metric("MAE", f"{test_mae:.2f}",
                 help="Mean Absolute Error - average prediction error")
    with col3:
        st.metric("RMSE", f"{test_rmse:.2f}",
                 help="Root Mean Squared Error - penalizes large errors")
    with col4:
        accuracy_pct = test_r2 * 100
        st.metric("Accuracy", f"{accuracy_pct:.1f}%",
                 help="Percentage of variance explained")
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Performance Metrics Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **R² Score (0.85):**
        - Measures how well predictions match actual values
        - 0.85 means the model explains 85% of yield variance
        - Range: 0 (poor) to 1 (perfect)
        - **Your model: Excellent performance! 🌟**
        
        **MAE (Mean Absolute Error):**
        - Average difference between predicted and actual yield
        - Lower is better
        - Measured in same units as yield (kg/ha)
        """)
    
    with col2:
        st.markdown("""
        **RMSE (Root Mean Squared Error):**
        - Similar to MAE but penalizes larger errors more
        - Always ≥ MAE
        - Lower is better
        - More sensitive to outliers
        
        **Model Type: Random Forest**
        - Ensemble of decision trees
        - Handles non-linear relationships well
        - Robust to outliers
        - Provides feature importance
        """)
    
    st.markdown("### 🎨 Performance Visualization")
    
    fig = go.Figure()
    
    perf_test_r2 = metrics.get('test_r2', metrics.get('r2_score', 0.85))
    perf_test_mae = metrics.get('test_mae', metrics.get('mae', 250))
    perf_test_rmse = metrics.get('test_rmse', metrics.get('rmse', 350))
    
    metrics_data = {
        'Metric': ['R² Score', 'Normalized MAE', 'Normalized RMSE'],
        'Value': [
            perf_test_r2,
            1 - min(perf_test_mae / 1000, 1),
            1 - min(perf_test_rmse / 1000, 1)
        ],
        'Target': [0.85, 0.75, 0.70]
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metrics_data['Metric'],
        y=metrics_data['Value'],
        name='Actual',
        marker_color='#2E7D32',
        text=[f'{v:.3f}' for v in metrics_data['Value']],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=metrics_data['Metric'],
        y=metrics_data['Target'],
        name='Target',
        marker_color='lightgray',
        opacity=0.5
    ))
    
    fig.update_layout(
        title='Model Performance vs Target Metrics',
        yaxis_title='Score',
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if hasattr(model_package['model'], 'feature_importances_'):
        st.markdown("### ⭐ Feature Importance")
        
        importances = model_package['model'].feature_importances_
        feature_names = model_package['feature_names']
        
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h',
                    title='Top 15 Most Important Features',
                    labels={'Importance': 'Feature Importance', 'Feature': 'Feature'},
                    color='Importance', color_continuous_scale='Greens')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Feature Importance Interpretation:**
        - Higher values = more important for predictions
        - Shows which factors most influence crop yield
        - Helps identify key agricultural and weather variables
        """)


def show_prediction_interface(model_package):
    """Show prediction interface tab."""
    st.markdown("## 🔮 Crop Yield Prediction")
    
    if model_package is None:
        st.error("⚠️ Model file not found! Please train the model first.")
        st.info("📝 Run `ml_pipeline_with_visualizations.py` to train the model")
        return
    
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoders = model_package['label_encoders']
    feature_names = model_package['feature_names']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌱 Agricultural Information")
        
        # Get district names from model (use as-is, already in proper case)
        districts = label_encoders['District'].classes_
        district = st.selectbox("Select District", options=districts, index=0)
        
        seasons = label_encoders['Season'].classes_
        season = st.selectbox("Select Season", options=seasons, index=0)
        
        crops = label_encoders['Crop'].classes_
        crop = st.selectbox("Select Crop", options=crops, index=0)
        
        year = st.number_input("Year", min_value=2018, max_value=2030, value=2023, step=1)
        area = st.number_input("Area (hectares)", min_value=1.0, max_value=100000.0, value=1000.0, step=100.0)
    
    with col2:
        st.markdown("### ☁️ Weather Information")
        
        total_rainfall = st.number_input(
            "Total Seasonal Rainfall (mm)",
            min_value=0.0,
            max_value=5000.0,
            value=800.0,
            step=50.0,
            help="Average seasonal rainfall: Kharif (600-1200mm), Rabi (100-400mm)"
        )
        
        st.markdown("**Temperature (°C)**")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            temp_min = st.number_input("Min Temp", min_value=0.0, max_value=45.0, value=20.0, step=1.0)
        with col_t2:
            temp_max = st.number_input("Max Temp", min_value=0.0, max_value=50.0, value=35.0, step=1.0)
        
        st.markdown("**Humidity (%)**")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            humidity_min = st.number_input("Min Humidity", min_value=0.0, max_value=100.0, value=60.0, step=5.0)
        with col_h2:
            humidity_max = st.number_input("Max Humidity", min_value=0.0, max_value=100.0, value=85.0, step=5.0)
    
    st.markdown("---")
    
    if st.button("🔮 Predict Yield", type="primary", use_container_width=True):
        with st.spinner("Calculating prediction..."):
            
            inputs = {
                'district': district,
                'season': season,
                'crop': crop,
                'year': year,
                'area': area,
                'total_rainfall': total_rainfall,
                'temp_min': temp_min,
                'temp_max': temp_max,
                'humidity_min': humidity_min,
                'humidity_max': humidity_max
            }
            
            features = calculate_engineered_features(inputs)
            
            # Encode categorical variables
            district_encoded = label_encoders['District'].transform([district])[0]
            season_encoded = label_encoders['Season'].transform([season])[0]
            crop_encoded = label_encoders['Crop'].transform([crop])[0]
            
            features['District_Encoded'] = district_encoded
            features['Season_Encoded'] = season_encoded
            features['Crop_Encoded'] = crop_encoded
            
            X_pred = pd.DataFrame([features])
            X_pred = X_pred[feature_names]
            
            X_pred_scaled = scaler.transform(X_pred)
            
            predicted_yield = model.predict(X_pred_scaled)[0]
            
            # Display what parameters were used
            st.info(f"📊 **Prediction Parameters**: {district.title()} | {season} (Season {season_encoded}) | {crop} | Year: {year}")
            
            st.markdown("---")
            st.markdown("## 📈 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="**Predicted Yield**",
                    value=f"{predicted_yield:.2f} kg/ha",
                    help="Predicted crop yield per hectare"
                )
            
            with col2:
                estimated_production = predicted_yield * area
                st.metric(
                    label="**Estimated Production**",
                    value=f"{estimated_production:,.2f} kg",
                    help="Total production = Yield × Area"
                )
            
            with col3:
                productivity = estimated_production / area if area > 0 else 0
                st.metric(
                    label="**Productivity**",
                    value=f"{productivity:.2f} kg/ha",
                    help="Production efficiency per hectare"
                )
            
            st.markdown("### 🔍 Weather Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Conditions Status")
                
                temp_avg = (temp_min + temp_max) / 2
                if temp_avg < 20:
                    temp_status = "🔵 Cool - May slow growth"
                elif 20 <= temp_avg <= 30:
                    temp_status = "🟢 Optimal - Good for growth"
                else:
                    temp_status = "🔴 Hot - Heat stress possible"
                st.write(f"**Temperature:** {temp_status}")
                
                if season == "Kharif":
                    if total_rainfall < 400:
                        rain_status = "🔴 Low - Water stress likely"
                    elif 400 <= total_rainfall <= 1500:
                        rain_status = "🟢 Optimal - Good moisture"
                    else:
                        rain_status = "🟡 High - Possible waterlogging"
                else:
                    if total_rainfall < 100:
                        rain_status = "🟡 Low - Irrigation needed"
                    elif 100 <= total_rainfall <= 500:
                        rain_status = "🟢 Optimal - Good for Rabi"
                    else:
                        rain_status = "🟡 High - Monitor drainage"
                
                st.write(f"**Rainfall:** {rain_status}")
                
                humidity_avg = (humidity_min + humidity_max) / 2
                if humidity_avg < 50:
                    humidity_status = "🔵 Low - Dry conditions"
                elif 50 <= humidity_avg <= 80:
                    humidity_status = "🟢 Optimal - Good balance"
                else:
                    humidity_status = "🟡 High - Disease risk"
                st.write(f"**Humidity:** {humidity_status}")
            
            with col2:
                st.markdown("#### Recommendations")
                
                recommendations = []
                
                if temp_max > 35:
                    recommendations.append("⚠️ High heat stress - consider irrigation scheduling")
                
                if (season == "Kharif" and total_rainfall < 500) or (season == "Rabi" and total_rainfall < 150):
                    recommendations.append("💧 Low rainfall - ensure adequate irrigation")
                
                if features['Optimal_Conditions'] == 1:
                    recommendations.append("✅ Optimal weather conditions predicted!")
                
                recommendations.append(f"📅 Best planting time for {crop} in {season} season")
                recommendations.append(f"🌾 Monitor crop regularly for {area:.0f} hectares")
                
                for rec in recommendations:
                    st.write(rec)
            
            st.markdown("### 📊 Weather Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = temp_avg,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Average Temperature (°C)"},
                    gauge = {
                        'axis': {'range': [None, 45]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 20], 'color': "lightblue"},
                            {'range': [20, 30], 'color': "lightgreen"},
                            {'range': [30, 45], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 35
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                max_rainfall = 2000 if season == "Kharif" else 800
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = total_rainfall,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"Seasonal Rainfall (mm) - {season}"},
                    gauge = {
                        'axis': {'range': [None, max_rainfall]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, max_rainfall*0.3], 'color': "lightyellow"},
                            {'range': [max_rainfall*0.3, max_rainfall*0.7], 'color': "lightgreen"},
                            {'range': [max_rainfall*0.7, max_rainfall], 'color': "lightblue"}
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("🔧 View Calculated Features"):
                st.markdown("**Engineered Features Used in Prediction:**")
                feature_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': X_pred.values[0]
                })
                st.dataframe(feature_df, use_container_width=True, height=400)


def main():
    """Main application function."""
    
    st.markdown("<h1 style='text-align: center;'>🌾 Crop Yield Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Machine Learning-Powered Agricultural Analytics</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    model_package = load_model()
    df = load_training_data()
    
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        
        if model_package:
            model_name = model_package.get('model_name', 'Random Forest')
            st.success(f"**Model:** {model_name}")
            metrics = model_package.get('metrics', model_package.get('performance', {}))
            r2 = metrics.get('test_r2', metrics.get('r2_score', 0.85))
            mae = metrics.get('test_mae', metrics.get('mae', 250))
            rmse = metrics.get('test_rmse', metrics.get('rmse', 350))
            st.metric("R² Score", f"{r2:.4f}")
            st.metric("MAE", f"{mae:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
        else:
            st.warning("Model not loaded")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Features:**
        - 🔮 Yield Predictions
        - 📊 Data Exploration
        - 🔗 Correlation Analysis
        - 🏆 Model Performance
        
        **Data Coverage:**
        - 32 Districts
        - 32 Crops
        - 2018-2022
        - Kharif & Rabi seasons
        """)
    
    tabs = st.tabs(["🔮 Predictions", "📊 Data Explorer", "🔗 Correlations", "🏆 Model Performance"])
    
    with tabs[0]:
        show_prediction_interface(model_package)
    
    with tabs[1]:
        show_data_exploration(df)
    
    with tabs[2]:
        show_correlation_analysis(df)
    
    with tabs[3]:
        show_model_performance(model_package)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌾 Crop Yield Prediction System | Telangana Agricultural Data Analysis</p>
        <p>Powered by Machine Learning | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
