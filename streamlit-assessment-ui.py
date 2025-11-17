import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Assessment Tool",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark teal theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #134e4a 50%, #0f172a 100%);
    }
    
    /* Headers */
    h1 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    h2, h3 {
        color: #5eead4 !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(20, 184, 166, 0.1) !important;
        border: 1px solid rgba(20, 184, 166, 0.3) !important;
        border-radius: 16px !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(20, 184, 166, 0.4) !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #f59e0b 0%, #fb923c 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(251, 146, 60, 0.3) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #d97706 0%, #f97316 100%) !important;
    }
    
    /* Radio buttons and selects */
    .stRadio, .stSelectbox {
        color: #5eead4 !important;
    }
    
    /* Sliders */
    .stSlider {
        padding: 10px 0 !important;
    }
    
    /* Text color */
    p, label, .stMarkdown {
        color: #99f6e4 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(20, 184, 166, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'assessment_mode' not in st.session_state:
    st.session_state.assessment_mode = 'General'
if 'feedback_level' not in st.session_state:
    st.session_state.feedback_level = 'A'
if 'grading_mode' not in st.session_state:
    st.session_state.grading_mode = 'Criterion-Referenced'
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Header
st.title("📝 Multi-Agent Written Assessment Tool")
st.markdown("**Configure assessment parameters and process submissions • CEFR A2/B1 Level**")
st.markdown("---")

# Assessment Mode Selection
st.subheader("Assessment Mode")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 General\n\nStandard assessment", use_container_width=True):
        st.session_state.assessment_mode = 'General'
        
with col2:
    if st.button("📚 Closed DLI\n\nCurriculum-based", use_container_width=True):
        st.session_state.assessment_mode = 'Closed DLI'
        
with col3:
    if st.button("📖 Open DLI\n\nCustom vocabulary", use_container_width=True):
        st.session_state.assessment_mode = 'Open DLI'

st.info(f"**Selected Mode:** {st.session_state.assessment_mode}")

# DLI Configuration
if st.session_state.assessment_mode == 'Closed DLI':
    st.markdown("---")
    st.subheader("📚 DLI Book Selection (Required)")
    dli_books = [f"DLI Book {i}" for i in range(11, 21)]
    selected_dli_book = st.selectbox(
        "Choose a DLI book",
        options=[''] + dli_books,
        index=0
    )
    if selected_dli_book:
        st.success(f"✓ Selected: {selected_dli_book}")

elif st.session_state.assessment_mode == 'Open DLI':
    st.markdown("---")
    st.subheader("📖 Custom DLI List (Required)")
    st.markdown("**Upload custom vocabulary/grammar list**")
    dli_file = st.file_uploader(
        "CSV format: term, type, difficulty, example",
        type=['csv'],
        key='dli_upload'
    )
    if dli_file:
        st.success(f"✓ Uploaded: {dli_file.name}")

# Feedback Detail Level
st.markdown("---")
st.subheader("Feedback Detail Level")
col1, col2, col3, col4 = st.columns(4)

feedback_options = {
    'A': 'Score only',
    'B': 'Brief feedback',
    'C': 'Detailed feedback',
    'D': 'Full breakdown'
}

with col1:
    if st.button("**A**\n\nScore only", use_container_width=True, key='fb_a'):
        st.session_state.feedback_level = 'A'
with col2:
    if st.button("**B**\n\nBrief feedback", use_container_width=True, key='fb_b'):
        st.session_state.feedback_level = 'B'
with col3:
    if st.button("**C**\n\nDetailed feedback", use_container_width=True, key='fb_c'):
        st.session_state.feedback_level = 'C'
with col4:
    if st.button("**D**\n\nFull breakdown", use_container_width=True, key='fb_d'):
        st.session_state.feedback_level = 'D'

st.info(f"**Selected Level:** Option {st.session_state.feedback_level} - {feedback_options[st.session_state.feedback_level]}")

# Grading Mode
st.markdown("---")
st.subheader("Grading Mode")
col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Criterion-Referenced\n\nFixed passing score", use_container_width=True, key='crit'):
        st.session_state.grading_mode = 'Criterion-Referenced'
        
with col2:
    if st.button("📈 Norm-Referenced\n\nRelative to batch", use_container_width=True, key='norm'):
        st.session_state.grading_mode = 'Norm-Referenced'

st.info(f"**Selected Mode:** {st.session_state.grading_mode}")

# Grading Threshold
st.markdown("---")
if st.session_state.grading_mode == 'Norm-Referenced':
    st.subheader("Passing Threshold (Norm-Referenced)")
    st.markdown("**Percentage of submissions receiving passing grade**")
    passing_percentage = st.slider(
        "Passing Percentage",
        min_value=0,
        max_value=100,
        value=70,
        step=1,
        label_visibility="collapsed"
    )
    st.markdown(f"### 🎯 {passing_percentage}%")
    
else:
    st.subheader("Passing Score (Criterion-Referenced)")
    st.markdown("**Minimum score required to pass**")
    criterion_threshold = st.slider(
        "Passing Score",
        min_value=0,
        max_value=100,
        value=60,
        step=1,
        label_visibility="collapsed"
    )
    st.markdown(f"### 🎯 {criterion_threshold}/100")

# File Upload
st.markdown("---")
st.subheader("Upload Submissions")
st.markdown("**CSV format: Student_ID, Question_Prompt, Student_Response**")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    key='submissions_upload',
    label_visibility="collapsed"
)

if uploaded_file:
    st.success(f"✓ Uploaded: {uploaded_file.name}")
    
    # Preview data
    try:
        df = pd.read_csv(uploaded_file)
        with st.expander("📋 Preview Submissions Data"):
            st.dataframe(df.head(), use_container_width=True)
            st.info(f"Total submissions: {len(df)}")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Action Buttons
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🚀 Run Assessment", use_container_width=True, type="primary"):
        # Validation
        errors = []
        
        if not uploaded_file:
            errors.append("⚠️ Please upload a submissions file")
            
        if st.session_state.assessment_mode == 'Closed DLI' and not selected_dli_book:
            errors.append("⚠️ Please select a DLI book")
            
        if st.session_state.assessment_mode == 'Open DLI' and not dli_file:
            errors.append("⚠️ Please upload a custom DLI list")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.session_state.processing = True
            
            # Prepare configuration summary
            grading_info = f"{passing_percentage}% norm-referenced" if st.session_state.grading_mode == 'Norm-Referenced' else f"{criterion_threshold}/100 criterion-referenced"
            
            st.success("✓ Validation passed!")
            st.info(f"""
            **Assessment Configuration:**
            - Mode: {st.session_state.assessment_mode}
            - Feedback Level: Option {st.session_state.feedback_level}
            - Grading: {grading_info}
            """)
            
            # Processing would happen here
            with st.spinner("🔄 Processing assessments... This may take up to 10 minutes."):
                # Placeholder for actual processing
                import time
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate processing
                for i in range(100):
                    time.sleep(0.05)  # Remove this in production
                    progress_bar.progress(i + 1)
                    status_text.text(f"Processing response {i+1}/100...")
                
                st.success("✅ Assessment complete!")
                st.balloons()

with col2:
    if st.button("⬇️ Download", use_container_width=True):
        st.info("Results download would trigger here")

# Configuration Summary
st.markdown("---")
st.subheader("⚙️ Assessment Configuration")

config_summary = f"""
**Mode:** {st.session_state.assessment_mode}

**Feedback Level:** Option {st.session_state.feedback_level} - {feedback_options[st.session_state.feedback_level]}

**Grading:** {st.session_state.grading_mode}
"""

if st.session_state.grading_mode == 'Norm-Referenced':
    config_summary += f"- Passing threshold: {passing_percentage if 'passing_percentage' in locals() else 70}% of submissions\n"
else:
    config_summary += f"- Passing score: {criterion_threshold if 'criterion_threshold' in locals() else 60}/100\n"

if st.session_state.assessment_mode == 'Closed DLI':
    config_summary += f"\n**DLI Book:** {selected_dli_book if 'selected_dli_book' in locals() and selected_dli_book else 'Not selected'}"
elif st.session_state.assessment_mode == 'Open DLI':
    config_summary += f"\n**Custom DLI List:** {dli_file.name if 'dli_file' in locals() and dli_file else 'Not uploaded'}"

config_summary += f"\n\n**Submissions:** {'Ready' if uploaded_file else 'No file uploaded'}"

st.info(config_summary)

# Footer
st.markdown("---")
st.caption("Multi-Agent Assessment System • CEFR A2/B1 Level • Powered by AI")