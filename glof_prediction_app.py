import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="GLOF Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
    }
    .risk-low {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_models():
    """Load all trained models and encoders"""
    try:
        models = {
            'risk_model': joblib.load('glof_risk_model.pkl'),
            'occurrence_model': joblib.load('glof_occurrence_model.pkl'),
            'scaler': joblib.load('glof_scaler.pkl'),
            'lake_type_encoder': joblib.load('lake_type_encoder.pkl'),
            'season_encoder': joblib.load('season_encoder.pkl')
        }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('synthetic_glof_dataset.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def predict_glof(parameters, models):
    """Make prediction using the trained models"""
    try:
        # Convert parameters to the right format
        row = parameters.copy()

        # Set season and days since 1990 to fixed values (since date fields are removed)
        # Use typical high-risk values or mean if needed
        row['Season_Encoded'] = 2  # e.g., Summer
        row['Days_Since_1990'] = 12000  # e.g., mid-2022

        # Risk indicators
        row['High_Altitude_Risk'] = int(row['Elevation_m'] > 4500)
        row['Large_Lake_Risk'] = int(row['Lake_Area_km2'] > 0.5)  # Approximate threshold
        row['High_Volume_Risk'] = int(row['Lake_Volume_MCM'] > 2.0)  # Approximate threshold
        row['Climate_Risk_Score'] = row['Temperature_Anomaly_C'] * row['PDD_Total'] / 1000
        row['Seismic_Risk_Score'] = row['Max_Earthquake_Magnitude'] / (row['Distance_to_Fault_km'] + 1)

        # Encode categorical variables
        try:
            row['Lake_Type_Encoded'] = models['lake_type_encoder'].transform([str(row['Lake_Type'])])[0]
        except:
            row['Lake_Type_Encoded'] = 0

        # Feature list (must match training features and order, no Year/Month/Day)
        features = [
            'Latitude', 'Longitude', 'Elevation_m',
            'Lake_Area_km2', 'Lake_Volume_MCM', 'Dam_Height_m', 'Dam_Width_m',
            'Temperature_Anomaly_C', 'Precipitation_mm', 'PDD_Total',
            'Seismic_Activity_Count', 'Max_Earthquake_Magnitude', 'Distance_to_Fault_km',
            'Glacier_Area_km2', 'Glacier_Retreat_Rate_m_per_year', 'Ice_Velocity_m_per_year',
            'Slope_Degree', 'Relief_m', 'Drainage_Basin_Area_km2', 'Previous_GLOF_Events',
            'Lake_Volume_per_Area', 'Dam_Aspect_Ratio', 'Glacier_to_Lake_Ratio',
            'High_Altitude_Risk', 'Large_Lake_Risk', 'High_Volume_Risk', 'Climate_Risk_Score', 'Seismic_Risk_Score',
            'Lake_Type_Encoded', 'Season_Encoded', 'Days_Since_1990'
        ]

        # Derived features
        row['Lake_Volume_per_Area'] = row['Lake_Volume_MCM'] / (row['Lake_Area_km2'] + 1e-6)
        row['Dam_Aspect_Ratio'] = row['Dam_Height_m'] / (row['Dam_Width_m'] + 1e-6)
        row['Glacier_to_Lake_Ratio'] = row['Glacier_Area_km2'] / (row['Lake_Area_km2'] + 1e-6)

        # Prepare features
        feat_values = []
        for feature in features:
            if feature in row:
                feat_values.append(row[feature])
            else:
                feat_values.append(0)

        feat = np.array(feat_values).reshape(1, -1)
        feat_scaled = models['scaler'].transform(feat)

    # ...

        # Make predictions
        occur_proba_raw = models['occurrence_model'].predict_proba(feat_scaled)
        occur_prob = float(occur_proba_raw[0][1])

        # Set risk level based on probability
        if occur_prob >= 0.66:
            risk_level = 2  # High
        elif occur_prob >= 0.33:
            risk_level = 1  # Medium
        else:
            risk_level = 0  # Low

    # ...

        return {
            'risk_level': risk_level,
            'occurrence_probability': occur_prob,
            'risk_level_text': ['Low', 'Medium', 'High'][risk_level]
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Home Page
def home_page():
    st.markdown('<h1 class="main-header">🌊 Glacial Lake Outburst Floods Prediction System</h1>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if models is None:
        st.error("Failed to load models. Please ensure all model files are present.")
        return
    
    # Load data for analysis
    df = load_data()
    if df is None:
        st.error("Failed to load data.")
        return
    
    # Key Metrics
    st.markdown("## 📊 Model Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Risk Level Accuracy</h3>
            <h2>70%</h2>
            <p>Multi-class Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>GLOF Detection</h3>
            <h2>99%</h2>
            <p>Binary Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Dataset Size</h3>
            <h2>1,000</h2>
            <p>Glacial Lakes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Features</h3>
            <h2>34</h3>
            <p>Environmental Factors</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Distribution
    st.markdown("## 🎯 Risk Level Distribution")
    
    risk_counts = df['GLOF_Risk_Level'].value_counts().sort_index()
    risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_labels,
        title="Distribution of Risk Levels",
        color_discrete_sequence=['#48dbfb', '#feca57', '#ff6b6b']
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # GLOF Occurrence
    st.markdown("## ⚠️ GLOF Occurrence Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        occur_counts = df['GLOF_Occurred'].value_counts()
        fig = px.bar(
            x=['No GLOF', 'GLOF Occurred'],
            y=occur_counts.values,
            title="GLOF Occurrence Count",
            color=['#48dbfb', '#ff6b6b'],
            color_discrete_map={}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top risk factors
        st.markdown("### 🔥 Top Risk Factors")
        risk_factors = [
            "Temperature Anomaly",
            "Lake Volume",
            "Climate Risk Score",
            "Earthquake Magnitude",
            "Glacier Retreat Rate"
        ]
        
        for i, factor in enumerate(risk_factors, 1):
            st.markdown(f"**{i}.** {factor}")
    
    # Quick Stats
    st.markdown("## 📈 Quick Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Elevation", f"{df['Elevation_m'].mean():.0f}m")
        st.metric("Average Lake Area", f"{df['Lake_Area_km2'].mean():.3f} km²")
    
    with col2:
        st.metric("Average Temperature Anomaly", f"{df['Temperature_Anomaly_C'].mean():.2f}°C")
        st.metric("Average Earthquake Magnitude", f"{df['Max_Earthquake_Magnitude'].mean():.2f}")
    
    with col3:
        st.metric("GLOF Occurrence Rate", f"{(df['GLOF_Occurred'].sum() / len(df) * 100):.1f}%")
        st.metric("High Risk Lakes", f"{(df['GLOF_Risk_Level'] == 2).sum()}")

# Analysis Page
def analysis_page():
    st.markdown('<h1 class="main-header">📊 Data Analysis & Insights</h1>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        st.error("Failed to load data.")
        return
    
    # Correlation Analysis
    st.markdown("## 🔗 Feature Correlation Analysis")
    
    # Select numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numerical_cols].corr()
    
    # Create correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("## 🎯 Feature Importance Analysis")
    
    # Simulated feature importance (replace with actual from your model)
    feature_importance = {
        'Temperature_Anomaly_C': 0.069,
        'Lake_Volume_MCM': 0.065,
        'Climate_Risk_Score': 0.064,
        'Max_Earthquake_Magnitude': 0.063,
        'Glacier_Retreat_Rate_m_per_year': 0.061,
        'Lake_Volume_per_Area': 0.059,
        'Precipitation_mm': 0.033,
        'Glacier_Area_km2': 0.032,
        'Lake_Area_km2': 0.031,
        'Dam_Width_m': 0.031
    }
    
    fig = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title="Top 10 Most Important Features",
        color=list(feature_importance.values()),
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution Analysis
    st.markdown("## 📈 Distribution Analysis")
    
    # Select key features for distribution analysis
    key_features = ['Elevation_m', 'Lake_Area_km2', 'Lake_Volume_MCM', 
                   'Temperature_Anomaly_C', 'Max_Earthquake_Magnitude']
    
    for feature in key_features:
        if feature in df.columns:
            fig = px.histogram(
                df, 
                x=feature, 
                title=f"Distribution of {feature}",
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk Analysis by Categories
    st.markdown("## ⚠️ Risk Analysis by Categories")
    
    # Lake Type Analysis
    lake_type_risk = df.groupby('Lake_Type')['GLOF_Risk_Level'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=lake_type_risk.index,
        y=lake_type_risk.values,
        title="Average Risk Level by Lake Type",
        color=lake_type_risk.values,
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Elevation Band Analysis
    if 'Elevation_Band' in df.columns:
        elevation_risk = df.groupby('Elevation_Band')['GLOF_Risk_Level'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=elevation_risk.index,
            y=elevation_risk.values,
            title="Average Risk Level by Elevation Band",
            color=elevation_risk.values,
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig, use_container_width=True)

# Prediction Page
def prediction_page():
    st.markdown('<h1 class="main-header">🔮 GLOF Prediction Tool</h1>', unsafe_allow_html=True)
    
    models = load_models()
    if models is None:
        st.error("Failed to load models. Please ensure all model files are present.")
        return
    
    st.markdown("### Enter the parameters for GLOF prediction:")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📍 Location")
            latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.0, format="%.6f")
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=86.9, format="%.6f")
            elevation = st.number_input("Elevation (m)", min_value=0.0, max_value=9000.0, value=4000.0)
            lake_type = st.selectbox("Lake Type", ["Moraine", "Ice", "Bedrock", "Mixed"], index=1)  # Ice (medium risk)
        
        with col2:
            st.markdown("#### 🏔️ Lake Characteristics")
            lake_area = st.number_input("Lake Area (km²)", min_value=0.001, max_value=50.0, value=2.5, format="%.6f")
            lake_volume = st.number_input("Lake Volume (MCM)", min_value=0.001, max_value=100.0, value=10.0, format="%.6f")
            dam_height = st.number_input("Dam Height (m)", min_value=1.0, max_value=200.0, value=60.0, format="%.2f")
            dam_width = st.number_input("Dam Width (m)", min_value=1.0, max_value=500.0, value=200.0, format="%.2f")
            
        
        st.markdown("#### 🌡️ Climate & Environmental")
        col3, col4 = st.columns(2)
        
        with col3:
            temp_anomaly = st.number_input("Temperature Anomaly (°C)", min_value=-10.0, max_value=10.0, value=2.5, format="%.2f")
            precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=1000.0, value=300.0, format="%.2f")
            pdd_total = st.number_input("PDD Total", min_value=0.0, max_value=10000.0, value=4000.0, format="%.2f")
        
        with col4:
            seismic_count = st.number_input("Seismic Activity Count", min_value=0, max_value=100, value=10)
            max_earthquake = st.number_input("Max Earthquake Magnitude", min_value=0.0, max_value=10.0, value=4.0, format="%.2f")
            fault_distance = st.number_input("Distance to Fault (km)", min_value=0.0, max_value=500.0, value=50.0, format="%.2f")
        
        st.markdown("#### 🧊 Glacier Characteristics")
        col5, col6 = st.columns(2)
        
        with col5:
            glacier_area = st.number_input("Glacier Area (km²)", min_value=0.0, max_value=1000.0, value=20.0, format="%.2f")
            glacier_retreat = st.number_input("Glacier Retreat Rate (m/year)", min_value=0.0, max_value=100.0, value=15.0, format="%.2f")
        
        with col6:
            ice_velocity = st.number_input("Ice Velocity (m/year)", min_value=0.0, max_value=1000.0, value=100.0, format="%.2f")
            slope_degree = st.number_input("Slope Degree", min_value=0.0, max_value=90.0, value=20.0, format="%.2f")
        
        st.markdown("#### 🏞️ Terrain & History")
        col7, col8 = st.columns(2)
        
        with col7:
            relief = st.number_input("Relief (m)", min_value=0.0, max_value=5000.0, value=1000.0, format="%.2f")
            drainage_area = st.number_input("Drainage Basin Area (km²)", min_value=0.0, max_value=10000.0, value=1500.0, format="%.2f")
        
        with col8:
            prev_glof = st.number_input("Previous GLOF Events", min_value=0, max_value=10, value=1)
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict GLOF Risk", use_container_width=True)
    
    if submitted:
        # Prepare parameters
        parameters = {
            'Latitude': latitude,
            'Longitude': longitude,
            'Elevation_m': elevation,
            'Lake_Area_km2': lake_area,
            'Lake_Volume_MCM': lake_volume,
            'Dam_Height_m': dam_height,
            'Dam_Width_m': dam_width,
            'Lake_Type': lake_type,
            'Temperature_Anomaly_C': temp_anomaly,
            'Precipitation_mm': precipitation,
            'PDD_Total': pdd_total,
            'Seismic_Activity_Count': seismic_count,
            'Max_Earthquake_Magnitude': max_earthquake,
            'Distance_to_Fault_km': fault_distance,
            'Glacier_Area_km2': glacier_area,
            'Glacier_Retreat_Rate_m_per_year': glacier_retreat,
            'Ice_Velocity_m_per_year': ice_velocity,
            'Slope_Degree': slope_degree,
            'Relief_m': relief,
            'Drainage_Basin_Area_km2': drainage_area,
            'Previous_GLOF_Events': prev_glof
        }
        
        # Make prediction
        with st.spinner("Making prediction..."):
            prediction = predict_glof(parameters, models)
        
        if prediction:
            # Display results
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                risk_class = prediction['risk_level_text']
                risk_color = {
                    'Low': 'risk-low',
                    'Medium': 'risk-medium', 
                    'High': 'risk-high'
                }.get(risk_class, 'risk-medium')
                
                st.markdown(f"""
                <div class="prediction-card {risk_color}">
                    <h2>Risk Level: {risk_class}</h2>
                    <h3>Score: {prediction['risk_level']}/2</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                prob = prediction['occurrence_probability']
                prob_color = 'risk-high' if prob > 0.5 else 'risk-medium' if prob > 0.2 else 'risk-low'
                
                st.markdown(f"""
                <div class="prediction-card {prob_color}">
                    <h2>GLOF Probability</h2>
                    <h3>{(prob * 100):.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk interpretation
            st.markdown("## 📋 Risk Interpretation")
            
            if prediction['risk_level'] == 0:
                st.success("✅ **Low Risk**: The glacial lake shows minimal risk factors for outburst flooding.")
            elif prediction['risk_level'] == 1:
                st.warning("⚠️ **Medium Risk**: The glacial lake shows some concerning factors. Monitor closely.")
            else:
                st.error("🚨 **High Risk**: The glacial lake shows significant risk factors. Immediate attention required!")
            
            # Probability interpretation
            if prediction['occurrence_probability'] < 0.1:
                st.info("💡 **Low Probability**: GLOF occurrence is unlikely in the near term.")
            elif prediction['occurrence_probability'] < 0.5:
                st.warning("⚠️ **Moderate Probability**: GLOF occurrence is possible. Enhanced monitoring recommended.")
            else:
                st.error("🚨 **High Probability**: GLOF occurrence is likely. Immediate action required!")

# Main App
def main():
    # Sidebar navigation
    st.sidebar.title("🌊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Analysis", "🔮 Prediction"]
    )
    
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Analysis":
        analysis_page()
    elif page == "🔮 Prediction":
        prediction_page()
    
    

if __name__ == "__main__":
    main()
