import streamlit as st
import pandas as pd
import numpy as np
from src.data_acquisition import SpaceXDataFetcher
from src.preprocessing import DataPreprocessor
from src.model import LaunchPredictor
from src.visualization import LaunchVisualizer
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="SpaceX Launch Analysis & Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS for beautiful design
css = '''
/* Base styles */
body {
    background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
    font-family: 'Inter', sans-serif;
}

.main {
    background: transparent !important;
    max-width: 100% !important;
    padding: 1rem !important;
}

/* Responsive container */
.stApp {
    max-width: 100% !important;
    padding: 0 !important;
}

/* Hero section */
.hero-section {
    background: linear-gradient(120deg, #232526 0%, #0f2027 100%);
    border-radius: 24px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 2rem;
    animation: fadeIn 1.2s;
}

@media (max-width: 768px) {
    .hero-section {
        flex-direction: column;
        text-align: center;
        padding: 1.5rem;
    }
}

.hero-title {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    color: #00eaff;
    margin-bottom: 0.5rem;
    letter-spacing: 1px;
    text-shadow: 0 2px 8px #00000055;
}

.hero-desc {
    font-size: clamp(1rem, 2vw, 1.2rem);
    color: #e0e0e0;
    margin-bottom: 0.5rem;
    line-height: 1.5;
}

.hero-img {
    border-radius: 18px;
    box-shadow: 0 4px 24px 0 #00000055;
    width: clamp(100px, 15vw, 120px);
    height: clamp(100px, 15vw, 120px);
    object-fit: cover;
    border: 2px solid #00eaff;
    background: #fff;
}

/* Sidebar */
.st-emotion-cache-1d391kg {
    background: linear-gradient(180deg, #232526 0%, #0f2027 100%) !important;
}

.st-emotion-cache-1v0mbdj {
    background: transparent !important;
}

.st-emotion-cache-1v0mbdj .stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    font-size: clamp(0.9rem, 1.5vw, 1.1rem);
    font-weight: 600;
    padding: 0.7rem 1.2rem;
    border-radius: 18px 18px 0 0;
    background: #23252622;
    color: #00eaff;
    transition: background 0.2s, color 0.2s;
    white-space: nowrap;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(120deg, #232526 0%, #0f2027 100%);
    border-radius: 18px;
    padding: clamp(1rem, 2vw, 1.5rem);
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px 0 #00000033;
    text-align: center;
    color: #00eaff;
    font-weight: 700;
    font-size: clamp(1.2rem, 2vw, 1.5rem);
    animation: fadeIn 1s;
    height: 100%;
}

/* Analysis sections */
.analysis-section {
    background: linear-gradient(120deg, #232526 0%, #0f2027 100%);
    border-radius: 18px;
    padding: clamp(1.5rem, 3vw, 2rem);
    margin: 1.5rem 0;
    box-shadow: 0 2px 12px 0 #00000033;
    animation: fadeIn 1s;
}

/* Plotly charts */
.js-plotly-plot {
    width: 100% !important;
    height: 100% !important;
}

/* Responsive grid */
.stTabs [role="tabpanel"] {
    padding: 1rem 0;
}

/* Form elements */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: #23252622 !important;
    border: 1px solid #00eaff33 !important;
    color: #fff !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(120deg, #00eaff 0%, #00a8ff 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    transition: transform 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #00eaff;
    border-radius: 4px;
}

::-webkit-scrollbar-track {
    background: #232526;
}

/* Responsive tables */
.stDataFrame {
    width: 100% !important;
    overflow-x: auto !important;
}

/* Map container */
.folium-map {
    width: 100% !important;
    height: 500px !important;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 1rem;
    color: #00eaff;
    font-size: 0.9rem;
}

/* Responsive columns */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
    }
    
    .metric-card {
        margin-bottom: 1rem;
    }
}
'''

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Sidebar logo
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/SpaceX-Logo.svg/320px-SpaceX-Logo.svg.png", width=180)

# Hero section
st.markdown(
    f"""
    <div class="hero-section">
        <img src="https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=crop&w=400&q=80" class="hero-img"/>
        <div>
            <div class="hero-title">🚀 SpaceX Launch Analysis & Prediction</div>
            <div class="hero-desc">A beautiful, interactive dashboard to explore, analyze, and predict SpaceX launches. Switch tabs for insights, trends, and predictions!</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
page = st.sidebar.radio(
    "Go to",
    ["Data Overview", "Launch Analysis", "Prediction Model", "Launch Sites Map"]
)

# Theme selector in sidebar
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.selectbox(
    "Select Theme",
    ["Modern Blue"],
    index=0
)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Title with animation
st.markdown('<h1 class="title-text">🚀 SpaceX Launch Analysis & Prediction Platform</h1>', unsafe_allow_html=True)

# Animated description
st.markdown("""
<div style='animation: fadeIn 1.5s ease-out; text-align: center; margin-bottom: 2em;'>
    This platform provides comprehensive analysis and prediction capabilities for SpaceX launches.
    Explore historical data, visualize trends, and predict the success of future launches.
</div>
""", unsafe_allow_html=True)

# Data fetching
@st.cache_data
def fetch_data():
    fetcher = SpaceXDataFetcher()
    return fetcher.fetch_all_data()

# Load and prepare data
@st.cache_data
def prepare_data():
    preprocessor = DataPreprocessor()
    launches_df = preprocessor.prepare_launches_data()
    launchpads_df = preprocessor.prepare_launchpads_data()
    rockets_df = preprocessor.prepare_rockets_data()
    
    # Merge data
    df = launches_df.merge(
        launchpads_df,
        left_on='launchpad',
        right_on='id',
        how='left',
        suffixes=('', '_launchpad')
    )
    
    df = df.merge(
        rockets_df,
        left_on='rocket',
        right_on='id',
        how='left',
        suffixes=('', '_rocket')
    )
    
    return df

# Train model
@st.cache_data
def train_model():
    """Train the model and return both the predictor and metrics."""
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_training_data()
    
    predictor = LaunchPredictor()
    metrics = predictor.train(X, y)
    
    return predictor, metrics

def show_prediction_model():
    """Show the prediction model interface."""
    st.title("Launch Success Prediction Model")
    
    # Train model
    predictor, metrics = train_model()
    
    # Display model metrics
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.2%}")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance = predictor.get_feature_importance()
    fig = px.bar(
        importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features'
    )
    st.plotly_chart(fig)
    
    # Prediction interface
    st.subheader("Make a Prediction")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic launch parameters
        year = st.number_input("Launch Year", min_value=2000, max_value=2024, value=2024)
        month = st.number_input("Launch Month", min_value=1, max_value=12, value=1)
        payload_mass = st.number_input("Payload Mass (kg)", min_value=0.0, value=1000.0)
        core_reused = st.checkbox("Core Reused")
        stages = st.number_input("Number of Stages", min_value=1, max_value=3, value=2)
        cost_per_launch = st.number_input("Cost per Launch (USD)", min_value=0, value=50000000)
        has_capsule = st.checkbox("Has Capsule")
        has_crew = st.checkbox("Has Crew")
        success_rate = st.slider("Historical Success Rate", 0.0, 1.0, 0.8)
    
    with col2:
        # Launch site parameters
        st.write("Launch Site Location")
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.5728)
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-80.6490)
        
        # Weather parameters
        st.write("Weather Conditions")
        temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=100.0, value=5.0)
        wind_direction = st.number_input("Wind Direction (degrees)", min_value=0, max_value=359, value=180)
        clouds = st.number_input("Cloud Coverage (%)", min_value=0, max_value=100, value=20)
        precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0)
        visibility = st.number_input("Visibility (m)", min_value=0, max_value=100000, value=10000)
        
        # Derived weather features
        is_windy = wind_speed > 20
        is_cloudy = clouds > 50
        is_rainy = precipitation > 0
        is_clear = visibility > 10000
        
        # Season
        season = pd.cut(
            [month],
            bins=[0, 3, 6, 9, 12],
            labels=['Winter', 'Spring', 'Summer', 'Fall']
        )[0]
    
    # Create prediction button
    if st.button("Predict Launch Success"):
        # Prepare input data with raw values
        raw_input = {
            'year': [year],
            'month': [month],
            'payload_mass': [payload_mass],
            'core_reused': [int(core_reused)],
            'latitude': [latitude],
            'longitude': [longitude],
            'stages': [stages],
            'cost_per_launch': [cost_per_launch],
            'has_capsule': [int(has_capsule)],
            'has_crew': [int(has_crew)],
            'success_rate': [success_rate],
            'temperature': [temperature],
            'humidity': [humidity],
            'wind_speed': [wind_speed],
            'wind_direction': [wind_direction],
            'clouds': [clouds],
            'precipitation': [precipitation],
            'visibility': [visibility],
            'is_windy': [int(is_windy)],
            'is_cloudy': [int(is_cloudy)],
            'is_rainy': [int(is_rainy)],
            'is_clear': [int(is_clear)],
            'season': [season]
        }
        
        # Create DataFrame with raw input
        input_data = pd.DataFrame(raw_input)
        
        # One-hot encode categorical features
        input_data = pd.get_dummies(input_data, columns=['season'], drop_first=True)
        
        # Make prediction
        prediction, probability = predictor.predict(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        success_prob = probability[0][1] * 100
        
        if prediction[0] == 1:
            st.success(f"Launch is predicted to be successful! (Confidence: {success_prob:.1f}%)")
        else:
            st.error(f"Launch is predicted to fail. (Confidence: {100-success_prob:.1f}%)")
        
        # Show probability distribution
        fig = go.Figure(data=[
            go.Bar(
                x=['Failure', 'Success'],
                y=[100-success_prob, success_prob],
                marker_color=['#e74c3c', '#2ecc71']
            )
        ])
        fig.update_layout(
            title="Prediction Probability Distribution",
            yaxis_title="Probability (%)",
            showlegend=False
        )
        st.plotly_chart(fig)
        
        # Show weather impact
        st.subheader("Weather Impact Analysis")
        weather_metrics = {
            'Temperature': temperature,
            'Humidity': humidity,
            'Wind Speed': wind_speed,
            'Cloud Coverage': clouds,
            'Precipitation': precipitation,
            'Visibility': visibility
        }
        
        # Create weather radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(weather_metrics.values()),
            theta=list(weather_metrics.keys()),
            fill='toself',
            name='Current Conditions'
        ))
        
        # Add ideal conditions
        ideal_conditions = {
            'Temperature': 25,
            'Humidity': 50,
            'Wind Speed': 5,
            'Cloud Coverage': 20,
            'Precipitation': 0,
            'Visibility': 10000
        }
        
        fig.add_trace(go.Scatterpolar(
            r=list(ideal_conditions.values()),
            theta=list(ideal_conditions.keys()),
            fill='toself',
            name='Ideal Conditions'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(weather_metrics.values()), max(ideal_conditions.values()))]
                )
            ),
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # Weather warnings
        st.subheader("Weather Warnings")
        warnings = []
        if is_windy:
            warnings.append("⚠️ High wind speed may affect launch")
        if is_cloudy:
            warnings.append("⚠️ High cloud coverage may affect visibility")
        if is_rainy:
            warnings.append("⚠️ Precipitation detected")
        if not is_clear:
            warnings.append("⚠️ Poor visibility conditions")
        
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("No weather warnings - conditions are favorable")

# Metric card function
def display_metric(label, value, delta=None):
    st.markdown(f"""
    <div class="metric-card">
        <div style='font-size:1.1rem; color:#fff; margin-bottom:0.2em;'>{label}</div>
        <div style='font-size:2.2rem; color:#00eaff;'>{value}</div>
        {f'<div style="font-size:1rem; color:#aaa;">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

# Main content
if page == "Data Overview":
    df = prepare_data()
    col1, col2, col3 = st.columns(3)
    with col1:
        display_metric("Total Launches", len(df))
    with col2:
        display_metric("Success Rate", f"{df['success'].mean():.1%}")
    with col3:
        display_metric("Unique Launch Sites", df['launchpad'].nunique())
    # Tabs for analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Launch Statistics", "📦 Payload Analysis", "📈 Success Trends", "📍 Site Performance"
    ])
    with tab1:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("Launch Statistics")
        fig = px.histogram(df, x='year', color='success',
                          title='Launch Distribution by Year',
                          labels={'year': 'Year', 'count': 'Number of Launches'},
                          color_discrete_sequence=['#00eaff', '#ff4444'])
        fig.update_layout(template='plotly_dark', font_color='#00eaff', title_font_size=22)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("Payload Analysis")
        fig = px.scatter(df, x='payload_mass', y='success',
                        title='Payload Mass vs Launch Success',
                        labels={'payload_mass': 'Payload Mass (kg)', 'success': 'Launch Success'},
                        color_discrete_sequence=['#00eaff'])
        fig.update_layout(template='plotly_dark', font_color='#00eaff', title_font_size=22)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab3:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("Success Trends")
        success_by_month = df.groupby('month')['success'].mean().reset_index()
        fig = px.line(success_by_month, x='month', y='success',
                     title='Success Rate by Month',
                     labels={'month': 'Month', 'success': 'Success Rate'},
                     color_discrete_sequence=['#00eaff'])
        fig.update_layout(template='plotly_dark', font_color='#00eaff', title_font_size=22)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab4:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("Site Performance")
        site_stats = df.groupby('full_name').agg({
            'success': ['count', 'mean']
        }).round(2)
        site_stats.columns = ['Total Launches', 'Success Rate']
        fig = px.bar(site_stats, y='Success Rate',
                    title='Success Rate by Launch Site',
                    color_discrete_sequence=['#00eaff'])
        fig.update_layout(template='plotly_dark', font_color='#00eaff', title_font_size=22)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Launch Analysis":
    df = prepare_data()
    tab1, tab2, tab3 = st.tabs([
        "📈 Success Rate Analysis", "📦 Payload Analysis", "⏳ Time Series Analysis"
    ])
    with tab1:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("Launch Success Rate Over Time")
        fig = px.line(
            df.groupby('year')['success'].mean().reset_index(),
            x='year',
            y='success',
            title='Launch Success Rate by Year',
            color_discrete_sequence=['#00eaff']
        )
        fig.update_layout(template='plotly_dark', font_color='#00eaff', title_font_size=22)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("Payload Mass Distribution")
        fig = px.box(
            df,
            x='success',
            y='payload_mass',
            title='Payload Mass Distribution by Launch Success',
            color_discrete_sequence=['#00eaff']
        )
        fig.update_layout(template='plotly_dark', font_color='#00eaff', title_font_size=22)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab3:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("Monthly Launch Distribution")
        monthly_launches = df.groupby(['year', 'month']).size().reset_index(name='count')
        fig = px.scatter(
            monthly_launches,
            x='month',
            y='year',
            size='count',
            title='Launch Distribution by Month and Year',
            color_discrete_sequence=['#00eaff']
        )
        fig.update_layout(template='plotly_dark', font_color='#00eaff', title_font_size=22)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Prediction Model":
    show_prediction_model()

elif page == "Launch Sites Map":
    df = prepare_data()
    st.header("Launch Sites Map")
    m = LaunchVisualizer().create_launch_sites_map(df)
    folium_static(m)
    st.subheader("Launch Site Statistics")
    site_stats = df.groupby('full_name').agg({
        'success': ['count', 'mean'],
        'payload_mass': 'mean'
    }).round(2)
    site_stats.columns = ['Total Launches', 'Success Rate', 'Avg Payload Mass']
    st.dataframe(site_stats, use_container_width=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 50px; color: #00eaff;'>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True) 