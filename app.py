"""
Mental Health Text Classifier - Streamlit Web Application

A beautiful, responsive web interface for mental health text classification
with dark theme, real-time predictions, and comprehensive analytics.

Author: [Your Name]
Date: July 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.preprocessing import preprocess_text, clean_text
from config.config import MODELS_DIR, STREAMLIT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"]
)

# Custom CSS for dark theme and modern UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom dark theme */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #8892b0;
        font-weight: 400;
    }
    
    /* Card styling */
    .prediction-card {
        background: rgba(100, 116, 139, 0.1);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 1rem;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metrics-card {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 0.75rem;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 1px #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 0.75rem;
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 20, 25, 0.8);
        backdrop-filter: blur(10px);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 0.75rem;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 0.75rem;
    }
    
    /* Animation classes */
    .fadeIn {
        animation: fadeIn 0.8s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load trained model artifacts with error handling."""
    try:
        model_path = MODELS_DIR / "tuned_lightgbm_model.pkl"
        vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        encoder_path = MODELS_DIR / "label_encoder.pkl"
        
        # Check if files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        logger.info(f"Loading vectorizer from: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        
        logger.info(f"Loading encoder from: {encoder_path}")
        label_encoder = joblib.load(encoder_path)
        
        logger.info("Successfully loaded all model artifacts")
        return model, vectorizer, label_encoder, True
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None, None, None, False


def predict_mental_health(text, model, vectorizer, label_encoder):
    """Make prediction on input text with confidence scores."""
    try:
        # Preprocess text
        clean_text_input = clean_text(text)
        
        if not clean_text_input.strip():
            return None, None, "Text is empty after preprocessing"
        
        # Vectorize
        X = vectorizer.transform([clean_text_input])
        
        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Get label
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence scores for all classes
        confidence_scores = dict(zip(label_encoder.classes_, probabilities))
        
        return predicted_label, confidence_scores, None
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None, None, str(e)


def create_confidence_chart(confidence_scores):
    """Create a beautiful confidence score visualization."""
    if not confidence_scores:
        return None
    
    labels = list(confidence_scores.keys())
    scores = list(confidence_scores.values())
    
    # Create color palette
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=scores,
            marker_color=colors[:len(labels)],
            text=[f'{score:.1%}' for score in scores],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Prediction Confidence Scores',
            'x': 0.5,
            'font': {'size': 18, 'color': '#f1f5f9'}
        },
        xaxis={
            'title': 'Mental Health Categories',
            'color': '#8892b0'
        },
        yaxis={
            'title': 'Confidence Score',
            'color': '#8892b0',
            'tickformat': '.0%'
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f1f5f9'},
        showlegend=False,
        height=400
    )
    
    return fig


def main():
    """Main application function."""
    
    # Header
    st.markdown("""
    <div class="main-header fadeIn">
        <h1 class="main-title">🧠 Mental Health Text Classifier</h1>
        <p class="main-subtitle">AI-powered analysis for mental health text classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model artifacts
    model, vectorizer, label_encoder, load_success = load_artifacts()
    
    if not load_success:
        st.error("""
        ❌ **Model Loading Error**
        
        Could not load the trained models. Please ensure:
        1. You have run the training pipeline (`python main.py`)
        2. Model files exist in the `models/` directory
        3. All required dependencies are installed
        """)
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses advanced machine learning to classify mental health-related text into different categories:
        
        - **Depression & Anxiety**
        - **Emotional Loneliness** 
        - **Panic & Family Issues**
        - **Trauma & Stress**
        
        The model was trained on curated mental health text data using state-of-the-art NLP techniques.
        """)
        
        st.markdown("### 🛠️ Model Info")
        if model and hasattr(model, 'feature_importances_'):
            st.metric("Model Type", "LightGBM")
            st.metric("Features", len(vectorizer.vocabulary_) if vectorizer else "N/A")
            st.metric("Categories", len(label_encoder.classes_) if label_encoder else "N/A")
        
        st.markdown("### � Usage Tips")
        st.markdown("""
        - **Be specific**: Provide detailed text for better accuracy
        - **Length matters**: 20-200 words work best
        - **Context helps**: Include emotional context
        - **Multiple sentences**: Use complete thoughts
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### � Enter Your Text")
        
        # Text input with placeholder
        text_input = st.text_area(
            "Type or paste your text here...",
            height=150,
            placeholder="Example: I've been feeling really overwhelmed lately with work and personal life. The stress is affecting my sleep and I find myself worrying constantly about things I can't control...",
            help="Enter text that you'd like to analyze for mental health classification."
        )
        
        # Prediction button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predict_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        
        if text_input:
            word_count = len(text_input.split())
            char_count = len(text_input)
            
            st.markdown(f"""
            <div class="metrics-card">
                <h4>📊 Text Metrics</h4>
                <p><strong>Words:</strong> {word_count}</p>
                <p><strong>Characters:</strong> {char_count}</p>
                <p><strong>Status:</strong> {'✅ Good length' if 20 <= word_count <= 200 else '⚠️ Consider 20-200 words'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Prediction results
    if predict_button and text_input.strip():
        with st.spinner("🤖 Analyzing your text..."):
            predicted_label, confidence_scores, error = predict_mental_health(
                text_input, model, vectorizer, label_encoder
            )
        
        if error:
            st.error(f"❌ **Prediction Error:** {error}")
        elif predicted_label and confidence_scores:
            # Main prediction result
            max_confidence = max(confidence_scores.values())
            
            st.markdown(f"""
            <div class="prediction-card fadeIn">
                <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Prediction Result</h3>
                <h2 style="color: #f1f5f9; margin-bottom: 0.5rem;">{predicted_label}</h2>
                <p style="color: #8892b0; font-size: 1.1rem;">Confidence: {max_confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence visualization
            if confidence_scores:
                st.plotly_chart(
                    create_confidence_chart(confidence_scores),
                    use_container_width=True
                )
            
            # Detailed breakdown
            st.markdown("### 📋 Detailed Analysis")
            
            for label, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{label}**")
                with col2:
                    st.write(f"{score:.1%}")
                st.progress(score)
                
            # Disclaimer
            st.markdown("""
            ---
            ⚠️ **Important Disclaimer**: This tool is for informational purposes only and should not replace professional mental health consultation. If you're experiencing mental health concerns, please consult with a qualified healthcare provider.
            """)
    
    elif predict_button and not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("""
    ---
    <div style="text-align: center; color: #8892b0; font-size: 0.9rem; padding: 2rem 0;">
        Made with ❤️ using Streamlit | Mental Health Text Classifier v1.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
