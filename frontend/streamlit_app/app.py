"""
Streamlit Frontend for AutoATC
AI-based Animal Type Classification system for cattle & buffaloes
"""

import streamlit as st
import requests
import json
import base64
import io
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="AutoATC - Animal Type Classification",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🐄 AutoATC - Animal Type Classification</h1>', unsafe_allow_html=True)
    st.markdown("**AI-based classification system for cattle & buffaloes under Rashtriya Gokul Mission**")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=AutoATC", width=200)
        
        st.markdown("### Navigation")
        page = st.selectbox(
            "Select Page",
            ["🏠 Home", "📊 Analysis", "📈 Results", "📤 Export", "⚙️ Settings", "📋 Validation"]
        )
        
        st.markdown("---")
        st.markdown("### System Status")
        if check_api_status():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Analysis":
        analysis_page()
    elif page == "📈 Results":
        results_page()
    elif page == "📤 Export":
        export_page()
    elif page == "⚙️ Settings":
        settings_page()
    elif page == "📋 Validation":
        validation_page()

def home_page():
    """Home page with system overview."""
    
    st.markdown("## Welcome to AutoATC")
    st.markdown("""
    AutoATC is an AI-powered system for automatic classification and scoring of cattle and buffaloes. 
    The system provides comprehensive analysis including:
    
    - **Animal Detection**: Identify cattle and buffaloes in images
    - **Breed Classification**: Classify specific breeds
    - **Body Measurements**: Calculate physical measurements
    - **ATC Scoring**: Generate Animal Type Classification scores
    - **Disease Detection**: Identify health issues
    - **BPA Integration**: Export to Bharat Pashudhan App
    """)
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analyses", "1,234", "12")
    
    with col2:
        st.metric("Success Rate", "94.2%", "2.1%")
    
    with col3:
        st.metric("Avg Processing Time", "2.3s", "-0.5s")
    
    with col4:
        st.metric("BPA Exports", "856", "23")
    
    # Recent analyses
    st.markdown("## Recent Analyses")
    
    # Get recent analyses (mock data for now)
    recent_data = get_recent_analyses()
    
    if recent_data:
        df = pd.DataFrame(recent_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent analyses found. Upload an image to get started!")

def analysis_page():
    """Analysis page for uploading and processing images."""

    st.markdown("## Image Analysis")

    # Backend connection test
    with st.expander("🔧 Backend Connection Test", expanded=False):
        if st.button("Test Backend Connection"):
            try:
                response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=5)
                if response.status_code == 200:
                    st.success(f"✅ Backend is running! {response.json()}")
                else:
                    st.error(f"❌ Backend returned status {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error(f"❌ Cannot connect to backend at {API_BASE_URL}")
                st.info("💡 Make sure to run: `cd AutoATC/backend && python simple_main.py`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image of a cattle or buffalo for analysis"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Analysis options
        st.markdown("### Analysis Options")

        col1, col2 = st.columns(2)

        with col1:
            include_breed = st.checkbox("Include Breed Classification", value=True)
            include_disease = st.checkbox("Include Disease Detection", value=True)

        with col2:
            include_measurements = st.checkbox("Include Body Measurements", value=True)
            animal_id = st.text_input("Animal ID (optional)", value="")

        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                result = analyze_image(
                    uploaded_file,
                    animal_id or None,
                    include_breed,
                    include_disease,
                    include_measurements
                )

                if result:
                    display_analysis_results(result)
                else:
                    st.warning("⚠️ Analysis failed. Check the error messages above.")

def results_page():
    """Results page for viewing analysis results."""
    
    st.markdown("## Analysis Results")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        animal_type = st.selectbox("Animal Type", ["All", "Cattle", "Buffalo"])
    
    with col2:
        breed = st.selectbox("Breed", ["All", "Gir", "Sahiwal", "Murrah", "Nili-Ravi"])
    
    with col3:
        date_range = st.date_input("Date Range", value=[datetime.now() - timedelta(days=30), datetime.now()])
    
    # Search
    search_term = st.text_input("Search by Animal ID", placeholder="Enter animal ID...")
    
    # Get results
    if st.button("🔍 Search Results"):
        with st.spinner("Loading results..."):
            results = get_analysis_results(animal_type, breed, date_range, search_term)
            
            if results:
                display_results_table(results)
            else:
                st.info("No results found matching your criteria.")

def export_page():
    """Export page for BPA integration."""
    
    st.markdown("## Export to BPA")
    
    # Animal selection
    animal_id = st.text_input("Animal ID", placeholder="Enter animal ID to export...")
    
    if animal_id:
        # Get animal info
        animal_info = get_animal_info(animal_id)
        
        if animal_info:
            st.markdown("### Animal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Animal ID:** {animal_info['animal_id']}")
                st.write(f"**Breed:** {animal_info.get('breed', 'Unknown')}")
                st.write(f"**ATC Score:** {animal_info.get('atc_score', 'N/A')}")
            
            with col2:
                st.write(f"**Analysis Date:** {animal_info.get('analysis_date', 'N/A')}")
                st.write(f"**Confidence:** {animal_info.get('confidence', 'N/A')}")
                st.write(f"**Status:** {animal_info.get('status', 'Unknown')}")
            
            # Export options
            st.markdown("### Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                include_measurements = st.checkbox("Include Measurements", value=True)
                include_diseases = st.checkbox("Include Disease Information", value=True)
            
            with col2:
                bpa_api_key = st.text_input("BPA API Key (optional)", type="password")
            
            # Export button
            if st.button("📤 Export to BPA", type="primary"):
                with st.spinner("Exporting to BPA..."):
                    result = export_to_bpa(animal_id, include_measurements, include_diseases, bpa_api_key)
                    
                    if result.get('success'):
                        st.success(f"✅ Successfully exported to BPA! BPA Animal ID: {result.get('bpa_animal_id', 'N/A')}")
                    else:
                        st.error(f"❌ Export failed: {result.get('message', 'Unknown error')}")
        else:
            st.error("Animal not found. Please check the Animal ID.")

def settings_page():
    """Settings page for system configuration."""
    
    st.markdown("## System Settings")
    
    # API Configuration
    st.markdown("### API Configuration")
    
    api_url = st.text_input("API Base URL", value=API_BASE_URL)
    
    if st.button("Test API Connection"):
        if check_api_status(api_url):
            st.success("✅ API connection successful!")
        else:
            st.error("❌ API connection failed!")
    
    # AI Model Settings
    st.markdown("### AI Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        detection_confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.1)
        breed_confidence = st.slider("Breed Confidence", 0.0, 1.0, 0.6, 0.1)
    
    with col2:
        keypoint_confidence = st.slider("Keypoint Confidence", 0.0, 1.0, 0.5, 0.1)
        disease_confidence = st.slider("Disease Confidence", 0.0, 1.0, 0.5, 0.1)
    
    # BPA Settings
    st.markdown("### BPA Integration Settings")
    
    bpa_api_url = st.text_input("BPA API URL", placeholder="https://api.bpa.gov.in/v1")
    bpa_api_key = st.text_input("BPA API Key", type="password")
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved successfully!")

def validation_page():
    """Validation page for accuracy assessment."""
    
    st.markdown("## Validation & Accuracy Assessment")
    
    # Validation form
    st.markdown("### Manual Validation")
    
    animal_id = st.text_input("Animal ID", placeholder="Enter animal ID to validate...")
    
    if animal_id:
        # Get AI results
        ai_results = get_animal_info(animal_id)
        
        if ai_results:
            st.markdown("#### AI Analysis Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**AI ATC Score:** {ai_results.get('atc_score', 'N/A')}")
                st.write(f"**AI Breed:** {ai_results.get('breed', 'Unknown')}")
            
            with col2:
                st.write(f"**AI Confidence:** {ai_results.get('confidence', 'N/A')}")
                st.write(f"**AI Measurements:** {ai_results.get('measurements', {})}")
            
            # Manual validation inputs
            st.markdown("#### Manual Validation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                manual_atc_score = st.number_input("Manual ATC Score", 0.0, 100.0, 0.0, 0.1)
                manual_breed = st.text_input("Manual Breed", placeholder="Enter actual breed...")
            
            with col2:
                validator_notes = st.text_area("Validator Notes", placeholder="Enter validation notes...")
            
            # Submit validation
            if st.button("✅ Submit Validation", type="primary"):
                with st.spinner("Submitting validation..."):
                    result = submit_validation(animal_id, manual_atc_score, manual_breed, validator_notes)
                    
                    if result.get('success'):
                        st.success("✅ Validation submitted successfully!")
                    else:
                        st.error(f"❌ Validation failed: {result.get('message', 'Unknown error')}")
        else:
            st.error("Animal not found. Please check the Animal ID.")
    
    # Accuracy report
    st.markdown("### Accuracy Report")
    
    if st.button("📊 Generate Accuracy Report"):
        with st.spinner("Generating accuracy report..."):
            report = get_accuracy_report()
            
            if report:
                display_accuracy_report(report)

# Helper functions

def check_api_status(api_url: str = API_BASE_URL) -> bool:
    """Check if API is accessible."""
    try:
        response = requests.get(f"{api_url}/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def analyze_image(uploaded_file, animal_id: str = None, include_breed: bool = True,
                 include_disease: bool = True, include_measurements: bool = True) -> dict:
    """Analyze uploaded image."""
    try:
        # Convert image to base64
        image_bytes = uploaded_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Prepare request data
        data = {
            "image_data": image_base64,
            "filename": uploaded_file.name,
            "animal_id": animal_id,
            "include_breed_classification": include_breed,
            "include_disease_detection": include_disease,
            "include_measurements": include_measurements
        }

        # Display debug info
        st.info(f"📡 Sending request to: {API_BASE_URL}/analyze")
        st.info(f"📦 Image size: {len(image_bytes)} bytes")

        # Make API request
        try:
            response = requests.post(f"{API_BASE_URL}/analyze", json=data, timeout=60)

            # Debug response
            st.info(f"📨 Response status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Analysis completed successfully!")
                return result
            else:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get('detail', response.text)
                except:
                    pass
                st.error(f"❌ Analysis failed (Status {response.status_code}): {error_detail}")
                return None

        except requests.exceptions.ConnectionError as e:
            st.error(f"❌ Connection Error: Cannot connect to backend at {API_BASE_URL}")
            st.error("Please ensure the backend is running on http://localhost:8000")
            st.code(f"Error details: {str(e)}")
            return None
        except requests.exceptions.Timeout as e:
            st.error(f"❌ Timeout Error: Request took longer than 60 seconds")
            st.code(f"Error details: {str(e)}")
            return None
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Request Error: {str(e)}")
            return None

    except Exception as e:
        st.error(f"❌ Error analyzing image: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def display_analysis_results(result: dict):
    """Display analysis results."""
    # Handle different response formats
    if 'data' in result:
        data = result.get('data', {})
    else:
        data = result

    st.success("✅ Analysis Complete!")

    # Debug: Show raw response structure
    with st.expander("🔍 Debug: Raw Response", expanded=False):
        st.json(result)

    # Basic information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Animal Type", data.get('animal_type', 'Unknown'))
        confidence = data.get('confidence', 0)
        if isinstance(confidence, (int, float)):
            st.metric("Confidence", f"{confidence:.2%}")
        else:
            st.metric("Confidence", str(confidence))

    with col2:
        atc_score_data = data.get('atc_score', {})
        score = atc_score_data.get('score', atc_score_data.get('overall_score', 'N/A'))
        st.metric("ATC Score", score)
        st.metric("ATC Grade", atc_score_data.get('grade', 'N/A'))

    with col3:
        breed_data = data.get('breed_classification', {})
        breed = breed_data.get('breed', breed_data.get('predicted_breed', 'Unknown'))
        st.metric("Breed", breed)
        processing_time = data.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")

    # Detailed results
    tabs = st.tabs(["📏 Measurements", "🏆 ATC Details", "🧬 Breed Info", "🏥 Health"])

    with tabs[0]:
        st.markdown("### Body Measurements")
        measurements = data.get('measurements', {})
        if measurements:
            # Display in a nice table format
            meas_data = []
            for key, value in measurements.items():
                if isinstance(value, (int, float)):
                    meas_data.append({"Measurement": key.replace('_', ' ').title(), "Value": f"{value:.2f}"})
                else:
                    meas_data.append({"Measurement": key.replace('_', ' ').title(), "Value": str(value)})
            if meas_data:
                st.table(pd.DataFrame(meas_data))
            else:
                st.json(measurements)
        else:
            st.info("No measurements available")

    with tabs[1]:
        st.markdown("### ATC Scoring Details")
        atc_score = data.get('atc_score', {})
        if atc_score:
            # Display factors
            factors = atc_score.get('factors', {})
            if factors:
                st.markdown("#### Scoring Factors")
                for factor, score in factors.items():
                    st.progress(score / 100, text=f"{factor.replace('_', ' ').title()}: {score:.1f}/100")

            # Display recommendations
            recommendations = atc_score.get('recommendations', [])
            if recommendations:
                st.markdown("#### Recommendations")
                for rec in recommendations:
                    st.write(f"• {rec}")

            # Show full data
            with st.expander("Full ATC Data"):
                st.json(atc_score)
        else:
            st.info("No ATC score available")

    with tabs[2]:
        st.markdown("### Breed Classification")
        breed_info = data.get('breed_classification', {})
        if breed_info:
            # Main breed
            main_breed = breed_info.get('breed', breed_info.get('predicted_breed', 'Unknown'))
            breed_conf = breed_info.get('confidence', 0)
            st.write(f"**Primary Breed:** {main_breed}")
            st.progress(breed_conf, text=f"Confidence: {breed_conf:.2%}")

            # Alternative breeds
            alternatives = breed_info.get('alternative_breeds', [])
            if alternatives:
                st.markdown("#### Alternative Breeds")
                for alt in alternatives:
                    breed_name = alt.get('breed', 'Unknown')
                    alt_conf = alt.get('confidence', 0)
                    st.write(f"• {breed_name}: {alt_conf:.2%}")

            # Show full data
            with st.expander("Full Breed Data"):
                st.json(breed_info)
        else:
            st.info("No breed information available")

    with tabs[3]:
        st.markdown("### Health & Disease Detection")
        # Check multiple possible keys for diseases
        diseases = data.get('diseases', [])
        disease_detection = data.get('disease_detection', {})
        if disease_detection:
            diseases = disease_detection.get('diseases_detected', diseases)
            health_score = disease_detection.get('health_score')
            if health_score:
                st.metric("Health Score", f"{health_score:.1f}/100")

            disease_recs = disease_detection.get('recommendations', [])
            if disease_recs:
                st.markdown("#### Recommendations")
                for rec in disease_recs:
                    st.write(f"• {rec}")

        if diseases:
            st.markdown("#### Detected Conditions")
            for disease in diseases:
                st.write(f"**{disease.get('name', 'Unknown')}** - Severity: {disease.get('severity', 'Unknown')}")
                if disease.get('description'):
                    st.write(disease.get('description', ''))
        else:
            st.success("✅ No diseases detected - Animal appears healthy!")

def get_recent_analyses() -> list:
    """Get recent analyses (mock data)."""
    return [
        {
            "Animal ID": "ATC001",
            "Type": "Cattle",
            "Breed": "Gir",
            "ATC Score": 85.2,
            "Date": "2024-01-15"
        },
        {
            "Animal ID": "ATC002",
            "Type": "Buffalo",
            "Breed": "Murrah",
            "ATC Score": 92.1,
            "Date": "2024-01-14"
        }
    ]

def get_analysis_results(animal_type: str, breed: str, date_range: list, search_term: str) -> list:
    """Get analysis results based on filters."""
    # This would make an API call to get results
    return []

def get_animal_info(animal_id: str) -> dict:
    """Get animal information."""
    try:
        response = requests.get(f"{API_BASE_URL}/results/{animal_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def export_to_bpa(animal_id: str, include_measurements: bool, include_diseases: bool, api_key: str = None) -> dict:
    """Export animal data to BPA."""
    try:
        data = {
            "animal_id": animal_id,
            "include_measurements": include_measurements,
            "include_diseases": include_diseases,
            "bpa_api_key": api_key
        }
        
        response = requests.post(f"{API_BASE_URL}/export/bpa", json=data)
        return response.json()
    except:
        return {"success": False, "message": "Export failed"}

def submit_validation(animal_id: str, manual_atc_score: float, manual_breed: str, notes: str) -> dict:
    """Submit manual validation."""
    try:
        data = {
            "animal_id": animal_id,
            "manual_atc_score": manual_atc_score,
            "manual_breed": manual_breed,
            "validator_notes": notes
        }
        
        response = requests.post(f"{API_BASE_URL}/validation", json=data)
        return response.json()
    except:
        return {"success": False, "message": "Validation failed"}

def get_accuracy_report() -> dict:
    """Get accuracy report."""
    try:
        response = requests.get(f"{API_BASE_URL}/results/accuracy-report")
        return response.json()
    except:
        return None

def display_results_table(results: list):
    """Display results in a table."""
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

def display_accuracy_report(report: dict):
    """Display accuracy report."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Validations", report.get('total_validations', 0))
        st.metric("ATC Score Accuracy", f"{report.get('atc_score_accuracy', 0):.2%}")
    
    with col2:
        st.metric("Breed Classification Accuracy", f"{report.get('breed_classification_accuracy', 0):.2%}")
        st.metric("Measurement Accuracy", f"{report.get('measurement_accuracy', 0):.2%}")
    
    # Recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        st.markdown("### Recommendations")
        for rec in recommendations:
            st.write(f"• {rec}")

if __name__ == "__main__":
    main()
