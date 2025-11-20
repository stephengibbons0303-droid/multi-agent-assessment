import streamlit as st
import pandas as pd
import os
from io import StringIO
import time
import asyncio
from assessment_agents import (
    LanguageControlAgent, CoherenceAgent, LexicalResourceAgent,
    TaskAchievementAgent, VerifierAgent
)
from dli_scanning_subagents import VocabularyScanningSub, GrammarScanningSub

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
    .stFileUploader label {
        color: #ffffff !important;
    }
    .stFileUploader section {
        color: #ffffff !important;
    }
    .stFileUploader small {
        color: #ffffff !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #f59e0b 0%, #fb923c 100%) !important;
        color: #112f38 !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px rgba(251, 146, 60, 0.3) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #d97706 0%, #f97316 100%) !important;
        color: #112f38 !important;
    }
    .stButton>button p {
        color: #112f38 !important;
    }
    .stButton>button span {
        color: #112f38 !important;
    }
    .stButton>button div {
        color: #112f38 !important;
    }
    .stButton button {
        color: #112f38 !important;
    }
    p, label, .stMarkdown { color: #99f6e4 !important; }
    .stButton p, .stButton label, .stButton .stMarkdown { color: #112f38 !important; }
    hr { border-color: rgba(20, 184, 166, 0.3) !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.5);
        border-radius: 8px 8px 0 0;
        color: #99f6e4;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(20, 184, 166, 0.2);
        color: #5eead4;
    }
    .stInfo {
        background-color: rgba(20, 184, 166, 0.15) !important;
    }
    .stInfo p {
        color: #ffffff !important;
    }
    .stSuccess {
        background-color: rgba(20, 184, 166, 0.15) !important;
    }
    .stSuccess p {
        color: #ffffff !important;
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
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'module_results_df' not in st.session_state:
    st.session_state.module_results_df = None

# API Key Configuration - Enhanced with Streamlit Secrets support
if 'api_key' not in st.session_state:
    try:
        st.session_state.api_key = st.secrets.get("OPENAI_API_KEY", "")
    except:
        st.session_state.api_key = os.getenv('OPENAI_API_KEY', '')

# Load DLI books data
@st.cache_data
def load_dli_books():
    """Load DLI book vocabulary data from CSV files"""
    dli_books = {}
    dli_dir = "dli_books"
    
    if os.path.exists(dli_dir):
        for i in range(11, 21):
            book_file = os.path.join(dli_dir, f"dli_book_{i}.csv")
            if os.path.exists(book_file):
                try:
                    df = pd.read_csv(book_file)
                    dli_books[f"DLI Book {i}"] = df.to_dict('records')
                except Exception as e:
                    st.warning(f"Could not load DLI Book {i}: {e}")
    
    return dli_books

# Sidebar for API key
with st.sidebar:
    st.title("⚙️ Configuration")
    
    if st.secrets.get("OPENAI_API_KEY"):
        st.success("✓ API Key loaded from secrets")
        st.info("💡 To update, modify `.streamlit/secrets.toml`")
    else:
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            help="Enter your OpenAI API key"
        )
        if api_key_input:
            st.session_state.api_key = api_key_input
    
    st.markdown("---")
    st.markdown("""
    **About this tool:**
    - CEFR A2/B1 Level Assessment
    - Multi-agent AI evaluation
    - DLI vocabulary & grammar tracking
    - Parallel processing (4x faster)
    - Module testing capability
    """)

# Header
st.title("📝 Multi-Agent Written Assessment Tool")
st.markdown("**CEFR A2/B1 Level • DLI-Enhanced Assessment • Module Testing**")

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
    st.info("💡 **Tip:** Set this in `.streamlit/secrets.toml` or as environment variable")
    st.markdown("---")
else:
    st.success("✓ API Key configured")
    st.markdown("---")

# Create tabs for Full Assessment and Module Testing
tab1, tab2 = st.tabs(["📊 Full Assessment", "🔬 Module Testing"])

# ============================================================================
# TAB 1: FULL ASSESSMENT
# ============================================================================

with tab1:
    st.subheader("Full Multi-Agent Assessment")
    st.markdown("Process complete assessments with all four criteria evaluated")
    
    # Assessment Mode Selection
    st.markdown("### Assessment Mode")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 General\n\nStandard assessment", use_container_width=True, key='tab1_general'):
            st.session_state.assessment_mode = 'General'
    with col2:
        if st.button("📚 Closed DLI\n\nCurriculum-based", use_container_width=True, key='tab1_closed'):
            st.session_state.assessment_mode = 'Closed DLI'
    with col3:
        if st.button("📖 Open DLI\n\nCustom vocabulary", use_container_width=True, key='tab1_open'):
            st.session_state.assessment_mode = 'Open DLI'
    
    st.info(f"**Selected Mode:** {st.session_state.assessment_mode}")
    
    # DLI Configuration
    selected_dli_book = None
    dli_file = None
    dli_items = []
    
    if st.session_state.assessment_mode == 'Closed DLI':
        st.markdown("---")
        st.markdown("### 📚 DLI Book Selection (Required)")
        
        available_books = load_dli_books()
        
        if available_books:
            book_options = [''] + list(available_books.keys())
            selected_dli_book = st.selectbox("Choose a DLI book", options=book_options, index=0, key='tab1_book')
            
            if selected_dli_book:
                st.success(f"✓ Selected: {selected_dli_book}")
                dli_items = available_books[selected_dli_book]
                
                with st.expander(f"📋 Preview {selected_dli_book} Vocabulary ({len(dli_items)} items)"):
                    preview_df = pd.DataFrame(dli_items)
                    st.dataframe(preview_df.head(10), use_container_width=True)
        else:
            st.error("⚠️ No DLI books found. Ensure data is in `dli_books/` directory.")
    
    elif st.session_state.assessment_mode == 'Open DLI':
        st.markdown("---")
        st.markdown("### 📖 Custom DLI List (Required)")
        dli_file = st.file_uploader(
            "CSV format: term, type, example",
            type=['csv'],
            key='tab1_dli_upload'
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
    st.markdown("### Feedback Detail Level")
    col1, col2, col3, col4 = st.columns(4)
    
    feedback_options = {
        'A': 'Score only',
        'B': 'Brief feedback',
        'C': 'Dtld feedback',
        'D': 'Full breakdown'
    }
    
    with col1:
        if st.button("**A**\n\nScore only", use_container_width=True, key='tab1_fb_a'):
            st.session_state.feedback_level = 'A'
    with col2:
        if st.button("**B**\n\nBrief feedback", use_container_width=True, key='tab1_fb_b'):
            st.session_state.feedback_level = 'B'
    with col3:
        if st.button("**C**\n\nDtld feedback", use_container_width=True, key='tab1_fb_c'):
            st.session_state.feedback_level = 'C'
    with col4:
        if st.button("**D**\n\nFull breakdown", use_container_width=True, key='tab1_fb_d'):
            st.session_state.feedback_level = 'D'
    
    st.info(f"**Selected Level:** Option {st.session_state.feedback_level} - {feedback_options[st.session_state.feedback_level]}")
    
    # Grading Mode
    st.markdown("---")
    st.markdown("### Grading Mode")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Criterion-Referenced\n\nFixed passing score", use_container_width=True, key='tab1_crit'):
            st.session_state.grading_mode = 'Criterion-Referenced'
    with col2:
        if st.button("📈 Norm-Referenced\n\nRelative to batch", use_container_width=True, key='tab1_norm'):
            st.session_state.grading_mode = 'Norm-Referenced'
    
    st.info(f"**Selected Mode:** {st.session_state.grading_mode}")
    
    # Grading Threshold
    st.markdown("---")
    passing_percentage = 70
    criterion_threshold = 60
    
    if st.session_state.grading_mode == 'Norm-Referenced':
        st.markdown("### Passing Threshold (Norm-Referenced)")
        passing_percentage = st.slider("Passing Percentage", 0, 100, 70, 1, key='tab1_norm_slider')
        st.markdown(f"#### 🎯 {passing_percentage}%")
    else:
        st.markdown("### Passing Score (Criterion-Referenced)")
        criterion_threshold = st.slider("Passing Score", 0, 100, 60, 1, key='tab1_crit_slider')
        st.markdown(f"#### 🎯 {criterion_threshold}/100")
    
    # File Upload
    st.markdown("---")
    st.markdown("### Upload Submissions")
    st.markdown("**CSV format: Student_ID, Question_Prompt, Student_Response**")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='tab1_submissions')
    
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
    
    # Processing Function with Sub-Agents
    async def assess_single_response_with_subagents(
        student_id, prompt, response, api_key, assessment_mode, 
        feedback_level, dli_items, agents, subagents
    ):
        """Assess a single response using parallel agent execution with sub-agent preprocessing"""
        
        lc_agent, coh_agent, lex_agent, task_agent, verifier = agents
        vocab_scanner, grammar_scanner = subagents
        
        try:
            # Run sub-agents first if DLI mode is active
            vocab_scan_results = None
            grammar_scan_results = None
            
            if assessment_mode in ['Closed DLI', 'Open DLI'] and dli_items:
                vocab_scan_results = vocab_scanner.scan(response, dli_items)
                grammar_scan_results = grammar_scanner.scan(response, dli_items)
            
            # Run all main agents in parallel
            results = await asyncio.gather(
                asyncio.to_thread(lc_agent.assess, response, feedback_level, grammar_scan_results),
                asyncio.to_thread(coh_agent.assess, response, feedback_level),
                asyncio.to_thread(lex_agent.assess, response, feedback_level, vocab_scan_results, assessment_mode),
                asyncio.to_thread(task_agent.assess, response, prompt, feedback_level),
                return_exceptions=True
            )
            
            lc_result, coh_result, lex_result, task_result = results
            
            # Check for exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    raise result
            
            # Collect scores for verification
            agent_scores = {
                'Language Control': lc_result.get('total_score', lc_result.get('score', 0)),
                'Coherence': coh_result['score'],
                'Lexical Resource': lex_result.get('total_score', lex_result.get('score', 0)),
                'Task Achievement': task_result['score']
            }
            
            # Verify scores
            verification = await asyncio.to_thread(
                verifier.verify, response, agent_scores, feedback_level
            )
            
            # Calculate weighted overall score
            # Language Control: 20% (16% base + 4% bonus)
            # Coherence: 20%
            # Lexical Resource: 20% (16% base + 4% bonus)
            # Task Achievement: 40%
            overall_score = (
                agent_scores['Language Control'] * 0.20 +
                agent_scores['Coherence'] * 0.20 +
                agent_scores['Lexical Resource'] * 0.20 +
                agent_scores['Task Achievement'] * 0.40
            )
            
            # Prepare result based on feedback level
            result = {
                'Student_ID': student_id,
                'Overall_Score': round(overall_score, 1)
            }
            
            if feedback_level == 'A':
                pass  # Just overall score
            
            elif feedback_level == 'B':
                combined_feedback = f"LC: {lc_result['feedback']}\n"
                combined_feedback += f"COH: {coh_result['feedback']}\n"
                combined_feedback += f"LEX: {lex_result['feedback']}\n"
                combined_feedback += f"TA: {task_result['feedback']}"
                result['Feedback'] = combined_feedback
            
            elif feedback_level == 'C':
                combined_feedback = f"Language Control: {lc_result['feedback']}\n\n"
                combined_feedback += f"Coherence: {coh_result['feedback']}\n\n"
                combined_feedback += f"Lexical Resource: {lex_result['feedback']}\n\n"
                combined_feedback += f"Task Achievement: {task_result['feedback']}"
                result['Feedback'] = combined_feedback
            
            elif feedback_level == 'D':
                result['Language_Control'] = lc_result.get('base_score', 0)
                result['LC_Bonus'] = lc_result.get('bonus_score', 0)
                result['Coherence'] = coh_result['score']
                result['Lexical_Resource'] = lex_result.get('base_score', 0)
                result['LEX_Bonus'] = lex_result.get('bonus_score', 0)
                result['Task_Achievement'] = task_result['score']
                if assessment_mode != 'General':
                    result['DLI_Vocab_Detected'] = lex_result.get('dli_items_detected', 0)
                result['Feedback'] = f"LC: {lc_result['feedback']}\n\nCOH: {coh_result['feedback']}\n\nLEX: {lex_result['feedback']}\n\nTA: {task_result['feedback']}"
                
                # Add bonus explanations if present
                if lc_result.get('bonus_explanation'):
                    result['LC_Bonus_Note'] = lc_result['bonus_explanation']
                if lex_result.get('bonus_explanation'):
                    result['LEX_Bonus_Note'] = lex_result['bonus_explanation']
            
            # Add anomaly info if detected
            if verification['anomalies_detected']:
                result['Anomaly_Flags'] = '; '.join(verification['anomalies'])
            
            return result, None
            
        except Exception as e:
            return None, str(e)
    
    def process_full_assessments(df, api_key, assessment_mode, feedback_level, dli_items, selected_dli_book):
        """Process all submissions through multi-agent assessment with sub-agents"""
        
        # Initialize agents
        lc_agent = LanguageControlAgent(api_key)
        coh_agent = CoherenceAgent(api_key)
        lex_agent = LexicalResourceAgent(api_key)
        task_agent = TaskAchievementAgent(api_key)
        verifier = VerifierAgent(api_key)
        
        # Initialize sub-agents
        vocab_scanner = VocabularyScanningSub()
        grammar_scanner = GrammarScanningSub()
        
        agents = (lc_agent, coh_agent, lex_agent, task_agent, verifier)
        subagents = (vocab_scanner, grammar_scanner)
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(df)
        errors = 0
        
        async def process_batch():
            nonlocal errors
            
            for idx, row in df.iterrows():
                student_id = row['Student_ID']
                prompt = row['Question_Prompt']
                response = row['Student_Response']
                
                status_text.text(f"Processing {idx+1}/{total}: Student {student_id}")
                
                # Retry logic
                max_retries = 3
                attempt = 0
                success = False
                
                while attempt < max_retries and not success:
                    try:
                        result, error = await assess_single_response_with_subagents(
                            student_id, prompt, response, api_key, 
                            assessment_mode, feedback_level, dli_items, agents, subagents
                        )
                        
                        if error:
                            raise Exception(error)
                        
                        results.append(result)
                        success = True
                        
                    except Exception as e:
                        attempt += 1
                        if attempt == max_retries:
                            results.append({
                                'Student_ID': student_id,
                                'Overall_Score': 'ERROR',
                                'Feedback': f'Processing failed after {max_retries} attempts'
                            })
                            errors += 1
                            status_text.text(f"⚠️ Error processing Student {student_id}: {str(e)}")
                            time.sleep(1)
                        else:
                            await asyncio.sleep(2 ** attempt)
                
                progress_bar.progress((idx + 1) / total)
        
        asyncio.run(process_batch())
        
        status_text.text(f"✅ Processing complete! {total - errors}/{total} successful")
        return pd.DataFrame(results), errors
    
    # Action Buttons
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Run Full Assessment", use_container_width=True, type="primary", key='tab1_run'):
            errors = []
            
            if not st.session_state.api_key:
                errors.append("⚠️ Please enter your OpenAI API key")
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
- Grading: {grading_info}
- DLI Scanning: {'Enabled' if st.session_state.assessment_mode != 'General' else 'Disabled'}""")
                
                with st.spinner("🔄 Processing full assessments with DLI scanning..."):
                    start_time = time.time()
                    
                    results_df, error_count = process_full_assessments(
                        submissions_df,
                        st.session_state.api_key,
                        st.session_state.assessment_mode,
                        st.session_state.feedback_level,
                        dli_items,
                        selected_dli_book
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    # Apply pass/fail
                    if st.session_state.grading_mode == 'Norm-Referenced':
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
                        st.warning(f"⚠️ {error_count} submission(s) failed processing")
                    
                    st.success(f"✅ Assessment complete in {elapsed_time:.1f} seconds!")
                    st.balloons()
    
    with col2:
        if st.button("⬇️ Download", use_container_width=True, key='tab1_download'):
            if st.session_state.results_df is not None:
                csv = st.session_state.results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="full_assessment_results.csv",
                    mime="text/csv",
                    key='tab1_download_btn'
                )
            else:
                st.info("No results to download yet")
    
    # Display Results
    if st.session_state.results_df is not None:
        st.markdown("---")
        st.markdown("### 📊 Assessment Results")
        st.dataframe(st.session_state.results_df, use_container_width=True)
        
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

# ============================================================================
# TAB 2: MODULE TESTING
# ============================================================================

with tab2:
    st.subheader("Module Testing")
    st.markdown("Test individual assessment agents in isolation")
    
    # Module Selection
    st.markdown("### Select Module to Test")
    module_options = {
        'Language Control': 'Grammar, syntax, and mechanics (16% base + 4% bonus)',
        'Coherence and Cohesion': 'Organization and logical flow (20%)',
        'Lexical Resource': 'Vocabulary range and precision (16% base + 4% bonus)',
        'Task Achievement': 'Response to prompt (40%)',
        'Verifier': 'Score consistency checking'
    }
    
    selected_module = st.selectbox(
        "Choose an agent to test:",
        options=list(module_options.keys()),
        format_func=lambda x: f"{x} - {module_options[x]}",
        key='tab2_module'
    )
    
    st.info(f"**Testing:** {selected_module}")
    
    # DLI Configuration for applicable modules
    tab2_dli_items = []
    tab2_assessment_mode = 'General'
    
    if selected_module in ['Language Control', 'Lexical Resource']:
        st.markdown("---")
        st.markdown("### DLI Configuration (Optional)")
        
        enable_dli = st.checkbox("Enable DLI scanning for this module", key='tab2_enable_dli')
        
        if enable_dli:
            dli_mode_choice = st.radio(
                "DLI Mode:",
                options=['Closed DLI', 'Open DLI'],
                key='tab2_dli_mode'
            )
            tab2_assessment_mode = dli_mode_choice
            
            if dli_mode_choice == 'Closed DLI':
                available_books = load_dli_books()
                if available_books:
                    book_options = [''] + list(available_books.keys())
                    tab2_selected_book = st.selectbox("Choose DLI book:", options=book_options, key='tab2_book')
                    if tab2_selected_book:
                        tab2_dli_items = available_books[tab2_selected_book]
                        st.success(f"✓ Loaded {len(tab2_dli_items)} items from {tab2_selected_book}")
            
            elif dli_mode_choice == 'Open DLI':
                tab2_dli_file = st.file_uploader("Upload custom DLI list:", type=['csv'], key='tab2_dli_file')
                if tab2_dli_file:
                    try:
                        dli_df = pd.read_csv(tab2_dli_file)
                        tab2_dli_items = dli_df.to_dict('records')
                        st.success(f"✓ Loaded {len(tab2_dli_items)} custom DLI items")
                    except Exception as e:
                        st.error(f"Error reading DLI file: {e}")
    
    # Feedback Level
    st.markdown("---")
    st.markdown("### Feedback Detail")
    tab2_feedback = st.radio(
        "Select feedback level:",
        options=['A', 'B', 'C', 'D'],
        format_func=lambda x: {'A': 'Score only', 'B': 'Brief', 'C': 'Detailed', 'D': 'Full'}[x],
        horizontal=True,
        key='tab2_feedback'
    )
    
    # File Upload
    st.markdown("---")
    st.markdown("### Upload Test Batch")
    st.markdown("**CSV format: Student_ID, Question_Prompt, Student_Response**")
    tab2_uploaded = st.file_uploader("Choose CSV file", type=['csv'], key='tab2_upload')
    
    tab2_submissions = None
    if tab2_uploaded:
        st.success(f"✓ Uploaded: {tab2_uploaded.name}")
        try:
            tab2_submissions = pd.read_csv(tab2_uploaded)
            with st.expander("📋 Preview Test Data"):
                st.dataframe(tab2_submissions.head(), use_container_width=True)
                st.info(f"Total submissions: {len(tab2_submissions)}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    # Process Module Testing
    def process_module_testing(df, api_key, selected_module, feedback_level, dli_items, assessment_mode):
        """Process batch through selected module only"""
        
        # Initialize the selected agent
        if selected_module == 'Language Control':
            agent = LanguageControlAgent(api_key)
            grammar_scanner = GrammarScanningSub() if dli_items else None
        elif selected_module == 'Coherence and Cohesion':
            agent = CoherenceAgent(api_key)
        elif selected_module == 'Lexical Resource':
            agent = LexicalResourceAgent(api_key)
            vocab_scanner = VocabularyScanningSub() if dli_items else None
        elif selected_module == 'Task Achievement':
            agent = TaskAchievementAgent(api_key)
        elif selected_module == 'Verifier':
            agent = VerifierAgent(api_key)
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total = len(df)
        errors = 0
        
        for idx, row in df.iterrows():
            student_id = row['Student_ID']
            prompt = row['Question_Prompt']
            response = row['Student_Response']
            
            status_text.text(f"Testing {idx+1}/{total}: Student {student_id}")
            
            try:
                if selected_module == 'Language Control':
                    grammar_scan = grammar_scanner.scan(response, dli_items) if grammar_scanner else None
                    result = agent.assess(response, feedback_level, grammar_scan)
                    
                    output = {
                        'Student_ID': student_id,
                        'Base_Score': result.get('base_score', result.get('score', 0)),
                        'Bonus_Score': result.get('bonus_score', 0),
                        'Total_Score': result.get('total_score', result.get('score', 0)),
                        'Feedback': result.get('feedback', '')
                    }
                    if result.get('bonus_explanation'):
                        output['Bonus_Note'] = result['bonus_explanation']
                
                elif selected_module == 'Lexical Resource':
                    vocab_scan = vocab_scanner.scan(response, dli_items) if vocab_scanner else None
                    result = agent.assess(response, feedback_level, vocab_scan, assessment_mode)
                    
                    output = {
                        'Student_ID': student_id,
                        'Base_Score': result.get('base_score', result.get('score', 0)),
                        'Bonus_Score': result.get('bonus_score', 0),
                        'Total_Score': result.get('total_score', result.get('score', 0)),
                        'DLI_Detected': result.get('dli_items_detected', 0),
                        'Feedback': result.get('feedback', '')
                    }
                    if result.get('bonus_explanation'):
                        output['Bonus_Note'] = result['bonus_explanation']
                
                elif selected_module in ['Coherence and Cohesion', 'Task Achievement']:
                    if selected_module == 'Coherence and Cohesion':
                        result = agent.assess(response, feedback_level)
                    else:
                        result = agent.assess(response, prompt, feedback_level)
                    
                    output = {
                        'Student_ID': student_id,
                        'Score': result['score'],
                        'Feedback': result.get('feedback', '')
                    }
                
                elif selected_module == 'Verifier':
                    # For verifier, we need mock scores to test
                    mock_scores = {
                        'Language Control': 15,
                        'Coherence': 16,
                        'Lexical Resource': 14,
                        'Task Achievement': 32
                    }
                    result = agent.verify(response, mock_scores, feedback_level)
                    
                    output = {
                        'Student_ID': student_id,
                        'Anomalies_Detected': result['anomalies_detected'],
                        'Score_Spread': result['score_spread'],
                        'Anomalies': '; '.join(result['anomalies']) if result['anomalies'] else 'None'
                    }
                
                results.append(output)
                
            except Exception as e:
                results.append({
                    'Student_ID': student_id,
                    'Error': str(e)
                })
                errors += 1
            
            progress_bar.progress((idx + 1) / total)
        
        status_text.text(f"✅ Module testing complete! {total - errors}/{total} successful")
        return pd.DataFrame(results), errors
    
    # Run Module Test Button
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🧪 Run Module Test", use_container_width=True, type="primary", key='tab2_run'):
            errors = []
            
            if not st.session_state.api_key:
                errors.append("⚠️ Please enter your OpenAI API key")
            if not tab2_uploaded:
                errors.append("⚠️ Please upload a test batch file")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                st.success("✓ Validation passed!")
                st.info(f"""**Module Test Configuration:**
- Module: {selected_module}
- Feedback Level: Option {tab2_feedback}
- DLI Mode: {tab2_assessment_mode}""")
                
                with st.spinner(f"🔄 Testing {selected_module} module..."):
                    start_time = time.time()
                    
                    module_results, error_count = process_module_testing(
                        tab2_submissions,
                        st.session_state.api_key,
                        selected_module,
                        tab2_feedback,
                        tab2_dli_items,
                        tab2_assessment_mode
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    st.session_state.module_results_df = module_results
                    
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} test(s) failed")
                    
                    st.success(f"✅ Module test complete in {elapsed_time:.1f} seconds!")
    
    with col2:
        if st.button("⬇️ Download", use_container_width=True, key='tab2_download'):
            if st.session_state.module_results_df is not None:
                csv = st.session_state.module_results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{selected_module.lower().replace(' ', '_')}_test_results.csv",
                    mime="text/csv",
                    key='tab2_download_btn'
                )
            else:
                st.info("No results to download yet")
    
    # Display Module Test Results
    if st.session_state.module_results_df is not None:
        st.markdown("---")
        st.markdown(f"### 🔬 {selected_module} Test Results")
        st.dataframe(st.session_state.module_results_df, use_container_width=True)
        
        # Module-specific statistics
        if 'Score' in st.session_state.module_results_df.columns:
            avg_score = st.session_state.module_results_df['Score'].mean()
            st.metric("Average Score", f"{avg_score:.1f}")
        elif 'Total_Score' in st.session_state.module_results_df.columns:
            avg_total = st.session_state.module_results_df['Total_Score'].mean()
            avg_bonus = st.session_state.module_results_df['Bonus_Score'].mean()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Total Score", f"{avg_total:.1f}")
            with col2:
                st.metric("Average Bonus", f"{avg_bonus:.1f}")

# Footer
st.markdown("---")
st.caption("Multi-Agent Assessment System • CEFR A2/B1 • DLI-Enhanced • Module Testing • v2.0")
