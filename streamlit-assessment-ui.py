import streamlit as st
import pandas as pd
import os
from io import StringIO
import time
from assessment_agents import (
    LanguageControlAgent, CoherenceAgent, LexicalResourceAgent,
    TaskAchievementAgent, VerifierAgent, DLIAgent
)

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Assessment Tool",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark teal theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #134e4a 50%, #0f172a 100%);
    }
    h1 { color: #ffffff !important; font-weight: 600 !important; }
    h2, h3 { color: #5eead4 !important; }
    .stAlert {
        background-color: rgba(20, 184, 166, 0.1) !important;
        border: 1px solid rgba(20, 184, 166, 0.3) !important;
        border-radius: 16px !important;
    }
    .stFileUploader {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(20, 184, 166, 0.4) !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }
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
    p, label, .stMarkdown { color: #99f6e4 !important; }
    hr { border-color: rgba(20, 184, 166, 0.3) !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'assessment_mode' not in st.session_state:
    st.session_state.assessment_mode = 'General'
if 'feedback_level' not in st.session_state:
    st.session_state.feedback_level = 'A'
if 'grading_mode' not in st.session_state:
    st.session_state.grading_mode = 'Criterion-Referenced'
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# API Key Configuration
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')

# Sidebar for API key
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter your OpenAI API key. You can also set it as OPENAI_API_KEY environment variable."
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
    
    st.markdown("---")
    st.markdown("""
    **About this tool:**
    - CEFR A2/B1 Level Assessment
    - Multi-agent AI evaluation
    - Option B verification (score consistency)
    - Auto-retry on failures
    """)

# Header
st.title("📝 Multi-Agent Written Assessment Tool")
st.markdown("**Configure assessment parameters and process submissions • CEFR A2/B1 Level**")

# Prominent API Key input at top if not set
if not st.session_state.api_key:
    st.warning("⚠️ OpenAI API Key Required")
    col1, col2 = st.columns([3, 1])
    with col1:
        main_api_key = st.text_input(
            "Enter your OpenAI API Key to get started:",
            type="password",
            placeholder="sk-...",
            key="main_api_key_input"
        )
        if main_api_key:
            st.session_state.api_key = main_api_key
            st.rerun()
    with col2:
        st.markdown("**[Get API Key →](https://platform.openai.com/api-keys)**")
    st.info("💡 **Tip:** You can also set this as an environment variable or use the sidebar (click `>` in top-left)")
    st.markdown("---")
else:
    st.success("✓ API Key configured")
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
selected_dli_book = None
dli_file = None
dli_items = []

if st.session_state.assessment_mode == 'Closed DLI':
    st.markdown("---")
    st.subheader("📚 DLI Book Selection (Required)")
    dli_books = [f"DLI Book {i}" for i in range(11, 21)]
    selected_dli_book = st.selectbox("Choose a DLI book", options=[''] + dli_books, index=0)
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
        try:
            dli_df = pd.read_csv(dli_file)
            dli_items = dli_df.to_dict('records')
            with st.expander("📋 Preview DLI Items"):
                st.dataframe(dli_df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading DLI file: {e}")

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
passing_percentage = 70
criterion_threshold = 60

if st.session_state.grading_mode == 'Norm-Referenced':
    st.subheader("Passing Threshold (Norm-Referenced)")
    st.markdown("**Percentage of submissions receiving passing grade**")
    passing_percentage = st.slider("Passing Percentage", 0, 100, 70, 1, label_visibility="collapsed")
    st.markdown(f"### 🎯 {passing_percentage}%")
else:
    st.subheader("Passing Score (Criterion-Referenced)")
    st.markdown("**Minimum score required to pass**")
    criterion_threshold = st.slider("Passing Score", 0, 100, 60, 1, label_visibility="collapsed")
    st.markdown(f"### 🎯 {criterion_threshold}/100")

# File Upload
st.markdown("---")
st.subheader("Upload Submissions")
st.markdown("**CSV format: Student_ID, Question_Prompt, Student_Response**")
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='submissions_upload', label_visibility="collapsed")

submissions_df = None
if uploaded_file:
    st.success(f"✓ Uploaded: {uploaded_file.name}")
    try:
        submissions_df = pd.read_csv(uploaded_file)
        with st.expander("📋 Preview Submissions Data"):
            st.dataframe(submissions_df.head(), use_container_width=True)
            st.info(f"Total submissions: {len(submissions_df)}")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Processing Function
def process_assessments(df, api_key, assessment_mode, feedback_level, dli_items, selected_dli_book):
    """Process all submissions through multi-agent assessment"""
    
    # Initialize agents
    lc_agent = LanguageControlAgent(api_key)
    coh_agent = CoherenceAgent(api_key)
    lex_agent = LexicalResourceAgent(api_key)
    task_agent = TaskAchievementAgent(api_key)
    verifier = VerifierAgent(api_key)
    dli_agent = DLIAgent(api_key) if assessment_mode != 'General' else None
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(df)
    errors = 0
    
    for idx, row in df.iterrows():
        student_id = row['Student_ID']
        prompt = row['Question_Prompt']
        response = row['Student_Response']
        
        status_text.text(f"Processing {idx+1}/{total}: Student {student_id}")
        
        # Retry logic for failed assessments
        max_retries = 3
        attempt = 0
        success = False
        
        while attempt < max_retries and not success:
            try:
                # Get DLI items for this assessment
                current_dli_items = []
                if assessment_mode == 'Closed DLI' and selected_dli_book:
                    # In production, load from pre-configured DLI books
                    current_dli_items = []  # Placeholder
                elif assessment_mode == 'Open DLI':
                    current_dli_items = [item['term'] for item in dli_items]
                
                # Run all agents in parallel (simulated sequential here)
                lc_result = lc_agent.assess(response, feedback_level)
                coh_result = coh_agent.assess(response, feedback_level)
                lex_result = lex_agent.assess(
                    response, feedback_level, 
                    current_dli_items, assessment_mode
                )
                task_result = task_agent.assess(response, prompt, feedback_level)
                
                # Collect scores for verification
                agent_scores = {
                    'Language Control': lc_result['score'],
                    'Coherence': coh_result['score'],
                    'Lexical Resource': lex_result['score'],
                    'Task Achievement': task_result['score']
                }
                
                # Verify scores
                verification = verifier.verify(response, agent_scores, feedback_level)
                
                # Calculate weighted overall score
                overall_score = (
                    lc_result['score'] * 0.20 +
                    coh_result['score'] * 0.20 +
                    lex_result['score'] * 0.20 +
                    task_result['score'] * 0.40
                )
                
                # Prepare result based on feedback level
                result = {
                    'Student_ID': student_id,
                    'Overall_Score': round(overall_score, 1)
                }
                
                if feedback_level == 'A':
                    pass  # Just overall score
                
                elif feedback_level == 'B':
                    combined_feedback = f"Language Control: {lc_result['feedback']}\n"
                    combined_feedback += f"Coherence: {coh_result['feedback']}\n"
                    combined_feedback += f"Lexical Resource: {lex_result['feedback']}\n"
                    combined_feedback += f"Task Achievement: {task_result['feedback']}"
                    result['Feedback'] = combined_feedback
                
                elif feedback_level == 'C':
                    combined_feedback = f"Language Control: {lc_result['feedback']}\n\n"
                    combined_feedback += f"Coherence: {coh_result['feedback']}\n\n"
                    combined_feedback += f"Lexical Resource: {lex_result['feedback']}\n\n"
                    combined_feedback += f"Task Achievement: {task_result['feedback']}"
                    result['Feedback'] = combined_feedback
                
                elif feedback_level == 'D':
                    result['Language_Control'] = lc_result['score']
                    result['Coherence'] = coh_result['score']
                    result['Lexical_Resource'] = lex_result['score']
                    result['Task_Achievement'] = task_result['score']
                    if assessment_mode != 'General':
                        result['DLI_Items_Used'] = lex_result.get('dli_items_used', 0)
                    result['Feedback'] = f"LC: {lc_result['feedback']}\n\nCOH: {coh_result['feedback']}\n\nLEX: {lex_result['feedback']}\n\nTA: {task_result['feedback']}"
                
                # Add anomaly info if detected
                if verification['anomalies_detected']:
                    result['Anomaly_Flags'] = '; '.join(verification['anomalies'])
                
                results.append(result)
                success = True
                
            except Exception as e:
                attempt += 1
                if attempt == max_retries:
                    # Failed after all retries
                    results.append({
                        'Student_ID': student_id,
                        'Overall_Score': 'ERROR',
                        'Feedback': f'Processing failed after {max_retries} attempts - please resubmit'
                    })
                    errors += 1
                    status_text.text(f"⚠️ Error processing Student {student_id}: {str(e)}")
                    time.sleep(1)
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        progress_bar.progress((idx + 1) / total)
    
    status_text.text(f"✅ Processing complete! {total - errors}/{total} successful")
    return pd.DataFrame(results), errors

# Action Buttons
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🚀 Run Assessment", use_container_width=True, type="primary"):
        errors = []
        
        if not st.session_state.api_key:
            errors.append("⚠️ Please enter your OpenAI API key in the sidebar")
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
            grading_info = f"{passing_percentage}% norm-referenced" if st.session_state.grading_mode == 'Norm-Referenced' else f"{criterion_threshold}/100 criterion-referenced"
            
            st.success("✓ Validation passed!")
            st.info(f"""**Assessment Configuration:**
- Mode: {st.session_state.assessment_mode}
- Feedback Level: Option {st.session_state.feedback_level}
- Grading: {grading_info}""")
            
            with st.spinner("🔄 Processing assessments... This may take up to 10 minutes."):
                results_df, error_count = process_assessments(
                    submissions_df,
                    st.session_state.api_key,
                    st.session_state.assessment_mode,
                    st.session_state.feedback_level,
                    dli_items,
                    selected_dli_book
                )
                
                # Apply pass/fail based on grading mode
                if st.session_state.grading_mode == 'Norm-Referenced':
                    # Sort and determine cutoff
                    valid_scores = results_df[results_df['Overall_Score'] != 'ERROR']['Overall_Score'].astype(float)
                    cutoff_index = int(len(valid_scores) * (1 - passing_percentage / 100))
                    sorted_scores = sorted(valid_scores, reverse=True)
                    cutoff_score = sorted_scores[cutoff_index] if cutoff_index < len(sorted_scores) else 0
                    results_df['Pass/Fail'] = results_df['Overall_Score'].apply(
                        lambda x: 'PASS' if x != 'ERROR' and float(x) >= cutoff_score else 'FAIL' if x != 'ERROR' else 'ERROR'
                    )
                else:
                    results_df['Pass/Fail'] = results_df['Overall_Score'].apply(
                        lambda x: 'PASS' if x != 'ERROR' and float(x) >= criterion_threshold else 'FAIL' if x != 'ERROR' else 'ERROR'
                    )
                
                st.session_state.results_df = results_df
                
                if error_count > 0:
                    st.warning(f"⚠️ {error_count} submission(s) failed processing. See results for details.")
                
                st.success("✅ Assessment complete!")
                st.balloons()

with col2:
    if st.button("⬇️ Download", use_container_width=True):
        if st.session_state.results_df is not None:
            csv = st.session_state.results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="assessment_results.csv",
                mime="text/csv"
            )
        else:
            st.info("No results to download yet. Run an assessment first.")

# Display Results
if st.session_state.results_df is not None:
    st.markdown("---")
    st.subheader("📊 Assessment Results")
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        total = len(st.session_state.results_df)
        st.metric("Total Submissions", total)
    with col2:
        passed = len(st.session_state.results_df[st.session_state.results_df['Pass/Fail'] == 'PASS'])
        st.metric("Passed", passed)
    with col3:
        failed = len(st.session_state.results_df[st.session_state.results_df['Pass/Fail'] == 'FAIL'])
        st.metric("Failed", failed)

# Configuration Summary
st.markdown("---")
st.subheader("⚙️ Assessment Configuration")

config_summary = f"""**Mode:** {st.session_state.assessment_mode}

**Feedback Level:** Option {st.session_state.feedback_level} - {feedback_options[st.session_state.feedback_level]}

**Grading:** {st.session_state.grading_mode}"""

if st.session_state.grading_mode == 'Norm-Referenced':
    config_summary += f"\n- Passing threshold: {passing_percentage}% of submissions"
else:
    config_summary += f"\n- Passing score: {criterion_threshold}/100"

if st.session_state.assessment_mode == 'Closed DLI':
    config_summary += f"\n\n**DLI Book:** {selected_dli_book if selected_dli_book else 'Not selected'}"
elif st.session_state.assessment_mode == 'Open DLI':
    config_summary += f"\n\n**Custom DLI List:** {dli_file.name if dli_file else 'Not uploaded'}"

config_summary += f"\n\n**Submissions:** {'Ready' if uploaded_file else 'No file uploaded'}"

st.info(config_summary)

st.markdown("---")
st.caption("Multi-Agent Assessment System • CEFR A2/B1 Level • Powered by GPT-4")
