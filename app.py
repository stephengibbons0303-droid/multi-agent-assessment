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

# Grading Context Helper Functions
def apply_grading_context(score, max_score, context_level):
    """
    Apply grading context adjustment using power curve.
    Higher context levels provide greater boost to lower scores.
    
    Args:
        score: Raw score (0-max_score)
        max_score: Maximum possible score
        context_level: Context level 0-10
    
    Returns:
        Adjusted score (0-max_score)
    """
    if score <= 0:
        return 0
    if score >= max_score:
        return max_score
    
    normalized = score / max_score
    exponent = 1.0 - (context_level * 0.05)  # 0: exp=1.0, 10: exp=0.5
    adjusted_normalized = normalized ** exponent
    adjusted_score = adjusted_normalized * max_score
    
    return min(adjusted_score, max_score)


def apply_feedback_tone_modifier(feedback_text, context_level, criterion_name):
    """
    Soften feedback language at high context levels (7-10).
    Maintains factual content while making tone more supportive.
    
    Args:
        feedback_text: Original feedback
        context_level: Grading context level 0-10
        criterion_name: Name of criterion being assessed
    
    Returns:
        Modified feedback text
    """
    if context_level < 7 or not feedback_text:
        return feedback_text
    
    # Word replacement patterns (harsh → supportive)
    replacements = {
        'frequent errors': 'areas for improvement',
        'numerous errors': 'several opportunities to develop',
        'many errors': 'several areas to work on',
        'fails to': 'is developing the ability to',
        'cannot': 'is learning to',
        'does not': 'is working toward',
        'poor': 'developing',
        'weak': 'emerging',
        'inadequate': 'foundational',
        'insufficient': 'developing',
        'needs significant improvement': 'has clear areas for growth',
        'needs improvement': 'has room to grow',
        'incorrectly': 'partially',
        'wrong': 'not yet accurate',
        'must': 'should',
        'lacks': 'would benefit from more practice with',
        'limited': 'developing',
        'very limited': 'foundational',
        'minimal': 'beginning',
        'basic': 'foundational',
        'struggled': 'found challenging',
        'difficulty': 'challenge',
        'unable to': 'working toward being able to',
        'failed to': 'did not yet',
        'severely': 'significantly',
        'serious': 'notable',
        'critical': 'important',
        'major': 'significant',
        'fundamental': 'important',
        'Below A2': 'Approaching A2',
        'not yet A2': 'approaching A2',
    }
    
    modified = feedback_text
    for harsh, supportive in replacements.items():
        modified = modified.replace(harsh, supportive)
    
    # Add context-appropriate framing
    if context_level >= 9:
        # Most supportive
        modified = f"In this practice context, your {criterion_name} demonstrates: {modified}"
    elif context_level >= 7:
        # Moderately supportive
        modified = f"{modified} Remember, this is a formative assessment focused on growth. These observations are meant to guide your continued development."
    
    return modified




# Initialize session state
if 'assessment_mode' not in st.session_state:
    st.session_state.assessment_mode = 'General'
if 'feedback_level' not in st.session_state:
    st.session_state.feedback_level = 'A'
if 'passing_threshold' not in st.session_state:
    st.session_state.passing_threshold = 60
if 'context_level' not in st.session_state:
    st.session_state.context_level = 5  # Default to balanced classwork
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'module_results_df' not in st.session_state:
    st.session_state.module_results_df = None

# API Key Configuration
# API Key Configuration
if 'api_key' not in st.session_state or not st.session_state.api_key:
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

# Sidebar
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
    - Grading context control
    - Parallel processing (4x faster)
    """)

# Header
st.title("📝 Written Assessment Tool")
st.markdown("**CEFR A2/B1 Level • DLI-Enhanced • Grading Context Control**")

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
    
    # Passing Threshold
    st.markdown("---")
    st.markdown("### Passing Threshold")
    st.markdown("Set the minimum score required to pass (criterion-referenced)")
    
    passing_threshold = st.slider(
        "Passing Score",
        min_value=0,
        max_value=100,
        value=st.session_state.passing_threshold,
        step=1,
        key='passing_threshold_slider',
        help="Students scoring at or above this threshold will pass"
    )
    st.session_state.passing_threshold = passing_threshold
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("50%", key="preset_50"):
            st.session_state.passing_threshold = 50
            st.rerun()
    with col2:
        if st.button("60%", key="preset_60"):
            st.session_state.passing_threshold = 60
            st.rerun()
    with col3:
        if st.button("70%", key="preset_70"):
            st.session_state.passing_threshold = 70
            st.rerun()
    with col4:
        if st.button("80%", key="preset_80"):
            st.session_state.passing_threshold = 80
            st.rerun()
    
    st.info(f"**Current threshold: {st.session_state.passing_threshold}/100**")
    
    # Grading Context Dial
    st.markdown("---")
    st.markdown("### Grading Context")
    st.markdown("Adjust scores based on assessment purpose and context")
    
    # Display context level definitions
    with st.expander("ℹ️ Context Level Guide"):
        st.markdown("""
**Context 0-2: High-Stakes/Summative**
- Final exams, certification, proficiency tests
- Minimal score adjustment (strict standards)
- Critical, objective feedback tone

**Context 3-6: Regular/Balanced** ⭐ *Recommended Default*
- Weekly assignments, classwork, homework
- Moderate score adjustment (~10-14 points)
- Constructive, balanced feedback tone

**Context 7-10: Formative/Diagnostic**
- Practice tests, drafts, diagnostic assessments
- Generous score adjustment (~16-22 points)
- Supportive, encouraging feedback tone
- Feedback language automatically softened

*Based on calibration data: AI scores ~18 points lower than human expert on average*
        """)
    
    # Grading context slider
    context_level = st.slider(
        "Context Level",
        min_value=0,
        max_value=10,
        value=st.session_state.context_level,
        step=1,
        key='context_slider',
        help="0=High-Stakes (strict), 5=Regular (balanced), 10=Formative (supportive)"
    )
    st.session_state.context_level = context_level
    
    # Quick preset buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("High-Stakes (0)", key="ctx_0"):
            st.session_state.context_level = 0
            st.rerun()
    with col2:
        if st.button("Mid-Term (4)", key="ctx_4"):
            st.session_state.context_level = 4
            st.rerun()
    with col3:
        if st.button("Classwork (5)", key="ctx_5"):
            st.session_state.context_level = 5
            st.rerun()
    with col4:
        if st.button("Diagnostic (8)", key="ctx_8"):
            st.session_state.context_level = 8
            st.rerun()
    
    # Display context description
    context_descriptions = {
        0: "**High-Stakes Testing** - Strict standards, minimal adjustment",
        1: "**Certification** - Very strict standards, objective evaluation",
        2: "**Final Exam** - Strict standards, formal assessment",
        3: "**Summative Assessment** - Moderate standards, balanced evaluation",
        4: "**Mid-Term** - Moderate-strict standards",
        5: "**Classwork** - Balanced standards (recommended default)",
        6: "**Formative** - Supportive standards, growth-focused",
        7: "**Progress Check** - Encouraging, formative feedback begins",
        8: "**Diagnostic** - Supportive, identifies gaps without penalty",
        9: "**Practice** - Very supportive, practice-oriented",
        10: "**Initial Screening** - Most supportive, exploratory"
    }
    
    st.info(f"**Level {st.session_state.context_level}:** {context_descriptions[st.session_state.context_level]}")
    
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
    
    # Processing Function
    async def assess_single_response_with_context(
        student_id, prompt, response, api_key, assessment_mode, 
        feedback_level, dli_items, agents, subagents, context_level
    ):
        """Assess a single response with grading context adjustment"""
        
        lc_agent, coh_agent, lex_agent, task_agent, verifier = agents
        vocab_scanner, grammar_scanner = subagents
        
        try:
            word_count = len(response.split())
            
            # Run sub-agents if DLI mode active
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
                asyncio.to_thread(task_agent.assess, response, prompt, feedback_level, word_count),
                return_exceptions=True
            )
            
            lc_result, coh_result, lex_result, task_result = results
            
            # Check for exceptions
            for result in results:
                if isinstance(result, Exception):
                    raise result
            
            # Extract raw scores
            lc_score = lc_result.get('total_score', lc_result.get('score', 0))
            coh_score = coh_result['score']
            lex_score = lex_result.get('total_score', lex_result.get('score', 0))
            task_score = task_result['score']
            
            # Calculate raw overall score (simple sum, not weighted)
            raw_overall = lc_score + coh_score + lex_score + task_score
            
            # Apply grading context adjustment
            adjusted_overall = apply_grading_context(raw_overall, 100, context_level)
            
            # Apply tone modifier to feedback if context >= 7
            if context_level >= 7 and feedback_level in ['B', 'C', 'D']:
                lc_result['feedback'] = apply_feedback_tone_modifier(
                    lc_result['feedback'], context_level, 'Language Control'
                )
                coh_result['feedback'] = apply_feedback_tone_modifier(
                    coh_result['feedback'], context_level, 'Coherence and Cohesion'
                )
                lex_result['feedback'] = apply_feedback_tone_modifier(
                    lex_result['feedback'], context_level, 'Lexical Resource'
                )
                task_result['feedback'] = apply_feedback_tone_modifier(
                    task_result['feedback'], context_level, 'Task Achievement'
                )
            
            # Collect scores for verification
            agent_scores = {
                'Language Control': lc_score,
                'Coherence': coh_score,
                'Lexical Resource': lex_score,
                'Task Achievement': task_score
            }
            
            # Verify scores
            verification = await asyncio.to_thread(
                verifier.verify, response, agent_scores, feedback_level
            )
            
            # Prepare result
            result = {
                'Student_ID': student_id,
                'Raw_Score': round(raw_overall, 1),
                'Overall_Score': round(adjusted_overall, 1)
            }
            
            if feedback_level == 'A':
                pass  # Just scores
            
            elif feedback_level == 'B':
                result['Language_Control'] = lc_score
                result['Coherence'] = coh_score
                result['Lexical_Resource'] = lex_score
                result['Task_Achievement'] = task_score
                
                combined_feedback = f"LC: {lc_result['feedback']}\n"
                combined_feedback += f"COH: {coh_result['feedback']}\n"
                combined_feedback += f"LEX: {lex_result['feedback']}\n"
                combined_feedback += f"TA: {task_result['feedback']}"
                result['Feedback'] = combined_feedback
            
            elif feedback_level == 'C':
                result['Language_Control'] = lc_score
                result['Coherence'] = coh_score
                result['Lexical_Resource'] = lex_score
                result['Task_Achievement'] = task_score
                
                combined_feedback = f"Language Control: {lc_result['feedback']}\n\n"
                combined_feedback += f"Coherence: {coh_result['feedback']}\n\n"
                combined_feedback += f"Lexical Resource: {lex_result['feedback']}\n\n"
                combined_feedback += f"Task Achievement: {task_result['feedback']}"
                result['Feedback'] = combined_feedback
            
            elif feedback_level == 'D':
                result['Language_Control_Base'] = lc_result.get('base_score', 0)
                result['LC_Bonus'] = lc_result.get('bonus_score', 0)
                result['LC_Total'] = lc_score
                result['Coherence'] = coh_score
                result['Lexical_Resource_Base'] = lex_result.get('base_score', 0)
                result['LEX_Bonus'] = lex_result.get('bonus_score', 0)
                result['LEX_Total'] = lex_score
                result['Task_Achievement'] = task_score
                
                if assessment_mode != 'General':
                    result['DLI_Vocab_Detected'] = lex_result.get('dli_items_detected', 0)
                
                result['Feedback'] = f"LC: {lc_result['feedback']}\n\nCOH: {coh_result['feedback']}\n\nLEX: {lex_result['feedback']}\n\nTA: {task_result['feedback']}"
                
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
    
    def process_full_assessments(df, api_key, assessment_mode, feedback_level, dli_items, selected_dli_book, context_level):
        """Process all submissions with grading context"""
        
        # Initialize agents
        lc_agent = LanguageControlAgent(api_key)
        coh_agent = CoherenceAgent(api_key)
        lex_agent = LexicalResourceAgent(api_key)
        task_agent = TaskAchievementAgent(api_key)
        verifier = VerifierAgent(api_key)
        
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
                
                max_retries = 3
                attempt = 0
                success = False
                
                while attempt < max_retries and not success:
                    try:
                        result, error = await assess_single_response_with_context(
                            student_id, prompt, response, api_key, 
                            assessment_mode, feedback_level, dli_items, agents, subagents, context_level
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
                                'Raw_Score': 'ERROR',
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
                st.success("✓ Validation passed!")
                st.info(f"""**Assessment Configuration:**
- Mode: {st.session_state.assessment_mode}
- Feedback Level: Option {st.session_state.feedback_level}
- Passing Threshold: {st.session_state.passing_threshold}/100
- Grading Context: Level {st.session_state.context_level}
- DLI Scanning: {'Enabled' if st.session_state.assessment_mode != 'General' else 'Disabled'}""")
                
                with st.spinner("🔄 Processing assessments with grading context adjustment..."):
                    start_time = time.time()
                    
                    results_df, error_count = process_full_assessments(
                        submissions_df,
                        st.session_state.api_key,
                        st.session_state.assessment_mode,
                        st.session_state.feedback_level,
                        dli_items,
                        selected_dli_book,
                        st.session_state.context_level
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    # Apply pass/fail based on adjusted scores
                    results_df['Pass/Fail'] = results_df['Overall_Score'].apply(
                        lambda x: 'PASS' if x != 'ERROR' and float(x) >= st.session_state.passing_threshold else 'FAIL' if x != 'ERROR' else 'ERROR'
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
                    file_name="assessment_results.csv",
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
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total = len(st.session_state.results_df)
            st.metric("Total Submissions", total)
        with col2:
            passed = len(st.session_state.results_df[st.session_state.results_df['Pass/Fail'] == 'PASS'])
            st.metric("Passed", passed)
        with col3:
            failed = len(st.session_state.results_df[st.session_state.results_df['Pass/Fail'] == 'FAIL'])
            st.metric("Failed", failed)
        with col4:
            if 'Raw_Score' in st.session_state.results_df.columns:
                valid_raw = st.session_state.results_df[st.session_state.results_df['Raw_Score'] != 'ERROR']['Raw_Score'].astype(float)
                valid_adjusted = st.session_state.results_df[st.session_state.results_df['Overall_Score'] != 'ERROR']['Overall_Score'].astype(float)
                avg_boost = valid_adjusted.mean() - valid_raw.mean()
                st.metric("Avg Context Boost", f"+{avg_boost:.1f}")

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
                        # Calculate word count for Task Achievement
                        word_count = len(response.split())
                        result = agent.assess(response, prompt, feedback_level, word_count)
                    
                    output = {
                        'Student_ID': student_id,
                        'Score': result['score'],
                        'Feedback': result.get('feedback', '')
                    }
                    
                    # Add word count to output for Task Achievement
                    if selected_module == 'Task Achievement':
                        output['Word_Count'] = word_count
                
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
st.caption("Multi-Agent Assessment System • CEFR A2/B1 • Grading Context Control • v3.0")
