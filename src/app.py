import streamlit as st
import os
import sys
from docx import Document

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.model_factory import ResumeModelFactory
from parsers.ats_parser import ATSParser
from parsers.industry_manager_parser import IndustryManagerParser
from evaluation.metrics import SummaryEvaluator
from config.app_config import UI_CONFIG

def load_css():
    st.markdown("""
        <style>
        /* CSS Variables for dark theme */
        :root {
            --primary: #60a5fa;
            --primary-light: #93c5fd;
            --primary-dark: #3b82f6;
            --secondary: #94a3b8;
            --accent: #38bdf8;
            --success: #4ade80;
            --success-light: rgba(74, 222, 128, 0.1);
            --warning: #fbbf24;
            --error: #f87171;
            --background: #0f172a;
            --surface: #1e293b;
            --surface-light: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --border: #334155;
            --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.4);
        }

        /* Global styles */
        .main {
            padding: 0.5rem;
            background-color: var(--background);
            max-width: 1200px;
            margin: 0 auto;
            color: var(--text-primary);
        }
        
        /* Modern card styling */
        .card {
            background: var(--surface);
            border-radius: 0.75rem;
            border: 1px solid var(--border);
            padding: 1rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }
        
        .card:hover {
            box-shadow: var(--shadow-md);
            border-color: var(--primary);
            transform: translateY(-1px);
        }
        
        /* Header styling */
        .header-container {
            background: linear-gradient(135deg, var(--surface), var(--surface-light));
            padding: 1.5rem;
            border-radius: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-md);
            text-align: center;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }
        
        .header-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, var(--primary-dark), var(--accent));
            opacity: 0.1;
            z-index: 0;
        }
        
        .header-container > * {
            position: relative;
            z-index: 1;
        }
        
        .stTitle {
            color: var(--primary-light);
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            margin: 0 !important;
            letter-spacing: -0.025em;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            font-weight: 400;
        }
        
        /* Step headers */
        .step-header {
            color: var(--primary-light);
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.5rem 0;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--surface);
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
        }
        
        .step-header::before {
            content: "";
            width: 3px;
            height: 1.5rem;
            background: var(--primary);
            border-radius: 1rem;
        }
        
        /* Section containers */
        .section-container {
            background: var(--surface);
            padding: 0.1rem;
            margin: 0.5rem 0;
            border-radius: 0.75rem;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }
        
        /* Form elements */
        .stRadio > label {
            color: var(--text-primary);
            font-weight: 500;
            margin-bottom: 0.25rem;
        }
        
        .stRadio > div {
            background: var(--surface-light);
            padding: 0.5rem;
            border-radius: 0.5rem;
            border: 1px solid var(--border);
        }
        
        .stRadio > div:hover {
            border-color: var(--primary);
        }
        
        .stButton > button {
            background: var(--primary);
            color: var(--text-primary);
            font-weight: 500;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            border: none;
            width: 100%;
            transition: all 0.2s ease;
            margin: 0.25rem 0;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        }
        
        .stButton > button:hover {
            background: var(--primary-light);
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        /* File uploader */
        .uploadedFile {
            border: 2px dashed var(--primary);
            border-radius: 0.75rem;
            padding: 1rem;
            text-align: center;
            background: var(--surface);
            margin: 0.5rem 0;
            color: var(--text-secondary);
        }
        
        /* Output styling */
        .output-container {
            background: var(--surface);
            padding: 1rem;
            border-radius: 0.75rem;
            margin: 0.5rem 0;
            border-left: 4px solid var(--success);
            border: 1px solid var(--border);
        }
        
        .output-container p {
            color: var(--text-primary);
            font-size: 1rem;
            line-height: 1.6;
            margin: 0;
        }
        
        /* Messages */
        .success-message {
            background: var(--success-light);
            color: var(--success);
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid rgba(74, 222, 128, 0.2);
        }
        
        .info-message {
            background: rgba(96, 165, 250, 0.1);
            color: var(--primary-light);
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid rgba(96, 165, 250, 0.2);
        }
        
        /* Selectbox */
        .stSelectbox {
            margin: 0.5rem 0;
        }
        
        .stSelectbox > div > div {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            color: var(--text-primary);
        }
        
        /* Help text */
        .stHelp {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Responsive grid */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        /* Utility classes */
        .text-sm { font-size: 0.875rem; }
        .text-lg { font-size: 1.125rem; }
        .font-medium { font-weight: 500; }
        .font-bold { font-weight: 700; }
        .text-center { text-align: center; }
        .mb-0 { margin-bottom: 0; }
        .mt-0 { margin-top: 0; }
        
        /* Dark theme specific overrides */
        .stTextInput > div > div {
            background: var(--surface);
            color: var(--text-primary);
            border-color: var(--border);
        }
        
        .stTextInput > div > div:hover {
            border-color: var(--primary);
        }
        
        .stMarkdown {
            color: var(--text-primary);
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--surface);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--surface-light);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary);
        }
        </style>
    """, unsafe_allow_html=True)

def read_docx_file(file):
    """Read and return the contents of an uploaded .docx file"""
    doc = Document(file)
    return doc

def main():
    st.set_page_config(
        page_title="Resume Summary Generator",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    load_css()
    
    # Initialize session state
    if 'generated_summary' not in st.session_state:
        st.session_state.generated_summary = ""
    if 'evaluation_scores' not in st.session_state:
        st.session_state.evaluation_scores = None
    
    # Modern header with gradient
    st.markdown("""
        <div class="header-container">
            <h1 class="stTitle">Resume Summary Generator</h1>
            <p class="subtitle">Transform your resume into a professional summary using AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns with better spacing
    col1, col2 = st.columns(UI_CONFIG['layout']['column_ratios']['main_content'], gap="large")
    
    with col1:
        with st.container():
            st.markdown('<h2 class="step-header">Step 1: Choose Template</h2>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                template_type = st.radio(
                    "Select template type:",
                    ["ATS Classic HR Resume", "Industry Manager Resume"],
                    label_visibility="collapsed"
                )
                
                # Template preview card
                st.markdown(f"""
                    <div class="card">
                        <p class="text-lg font-medium mb-0">{template_type}</p>
                        <p class="text-sm text-secondary mt-0">Perfect for {
                            "HR and recruitment positions" if template_type == "ATS Classic HR Resume" 
                            else "senior management roles"
                        }</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # File paths
                base_path = "src/templates"
                template_paths = {
                    "ATS Classic HR Resume": os.path.join(base_path, "ATS classic HR resume.docx"),
                    "Industry Manager Resume": os.path.join(base_path, "Industry manager resume.docx")
                }
                
                if os.path.exists(template_paths[template_type]):
                    with open(template_paths[template_type], "rb") as file:
                        template_content = file.read()
                        st.download_button(
                            label=f"📥 Download {template_type}",
                            data=template_content,
                            file_name=os.path.basename(template_paths[template_type]),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="template_download"
                        )
                else:
                    st.error("⚠️ Template file not found!")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # File Upload Section
        st.markdown('<h2 class="step-header">Step 2: Upload Resume</h2>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.write("Fill out the template and upload your resume:")
            uploaded_file = st.file_uploader(
                "Drop your .docx file here",
                type=["docx"],
                help="Only .docx files are supported"
            )
            if uploaded_file:
                st.success("✅ File uploaded successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Model Selection
        st.markdown('<h2 class="step-header">Step 3: Select AI Model</h2>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            
            selected_model = st.selectbox(
                "Choose your preferred AI model:",
                list(UI_CONFIG['models'].keys()),
                help="Each model has its own strengths"
            )
            
            st.info(f"💡 {UI_CONFIG['models'][selected_model]['description']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate Summary Button (Full Width)
    st.markdown('<h2 class="step-header">Generate Summary</h2>', unsafe_allow_html=True)
    if uploaded_file is not None and st.button("🚀 Generate Professional Summary"):
        with st.spinner("🔄 Processing your resume..."):
            try:
                # Parse resume based on template
                if template_type == "ATS Classic HR Resume":
                    parser = ATSParser(uploaded_file)
                else:
                    parser = IndustryManagerParser(uploaded_file)
                
                input_data = parser.parse_docx_to_json()
                
                # Show parsed data in expander
                with st.expander("🔍 View Parsed Resume Data"):
                    st.json(input_data)
                
                # Generate summary using the selected model
                factory = ResumeModelFactory()
                model_config = UI_CONFIG['models'][selected_model]
                model = factory.create_model(model_config['type'], model_config['size'])
                st.session_state.generated_summary = model.generate_summary(input_data)
                
                # Evaluate the generated summary
                evaluator = SummaryEvaluator()
                reference_summary = input_data.get('reference_summary', '')  # Get reference if available
                st.session_state.evaluation_scores = evaluator.evaluate_summary(
                    st.session_state.generated_summary,
                    reference_summary,
                    input_data
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if st.checkbox("Show detailed error"):
                    st.exception(e)
    elif uploaded_file is None and st.button("🚀 Generate Professional Summary"):
        st.warning("⚠️ Please upload your resume first!")
    
    # Display summary section (always show if there's a generated summary)
    if st.session_state.generated_summary:
        st.markdown('<h3 style="color: #4CAF50; margin-top: 2rem;">✨ Generated Summary</h3>', unsafe_allow_html=True)
        
        # Add custom CSS for the text area
        text_area_config = UI_CONFIG['layout']['text_area']
        st.markdown(f"""
            <style>
            .stTextArea textarea {{
                font-family: {text_area_config['font_family']};
                font-size: {text_area_config['font_size']};
                line-height: {text_area_config['line_height']};
                padding: 1rem;
                background-color: var(--surface);
                color: var(--text-primary);
                border: 1px solid var(--border);
                border-radius: 0.5rem;
                min-height: {text_area_config['min_height']}px;
            }}
            .stTextArea textarea:focus {{
                border-color: var(--primary);
                box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2);
            }}
            .metric-card {{
                background: var(--surface);
                padding: 1rem;
                border-radius: 0.5rem;
                border: 1px solid var(--border);
                margin: 0.5rem 0;
            }}
            .metric-value {{
                font-size: 1.5rem;
                font-weight: bold;
                color: var(--primary);
            }}
            .metric-label {{
                color: var(--text-secondary);
                font-size: 0.875rem;
            }}
            </style>
        """, unsafe_allow_html=True)
        
        # Editable text area for the summary
        edited_summary = st.text_area(
            "Edit your summary:",
            value=st.session_state.generated_summary,
            height=text_area_config['default_height'],
            label_visibility="collapsed"
        )
        
        # Display evaluation metrics if available
        if st.session_state.evaluation_scores:
            st.markdown('<h3 style="margin-top: 2rem;">📊 Quality Metrics</h3>', unsafe_allow_html=True)
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.markdown("#### Content Quality")
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.2f}</div>
                        <div class="metric-label">Content Similarity</div>
                    </div>
                """.format(st.session_state.evaluation_scores['content_similarity']), unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.2f}</div>
                        <div class="metric-label">ROUGE-1 Score</div>
                    </div>
                """.format(st.session_state.evaluation_scores['rouge1_f']), unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.2f}</div>
                        <div class="metric-label">ROUGE-L Score</div>
                    </div>
                """.format(st.session_state.evaluation_scores['rougeL_f']), unsafe_allow_html=True)
            
            with metrics_col2:
                st.markdown("#### Readability Metrics")
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.1f}</div>
                        <div class="metric-label">Avg. Sentence Length</div>
                    </div>
                """.format(st.session_state.evaluation_scores['avg_sentence_length']), unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.2f}</div>
                        <div class="metric-label">Sentiment Score</div>
                        <div class="metric-desc">(-1 negative to +1 positive)</div>
                    </div>
                """.format(st.session_state.evaluation_scores['sentiment_score']), unsafe_allow_html=True)
                
                st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.2f}</div>
                        <div class="metric-label">Objectivity Score</div>
                        <div class="metric-desc">(0 subjective to 1 objective)</div>
                    </div>
                """.format(1 - st.session_state.evaluation_scores['subjectivity_score']), unsafe_allow_html=True)
        
        # Action buttons in columns
        col1, col2, col3 = st.columns(UI_CONFIG['layout']['column_ratios']['action_buttons'])
        with col1:
            if st.button("📋 Copy to Clipboard"):
                st.write('<script>navigator.clipboard.writeText(`' + edited_summary + '`);</script>', unsafe_allow_html=True)
                st.success("✅ Copied to clipboard!")
        
        with col2:
            # Download button for the edited summary
            st.download_button(
                label="📥 Download Summary",
                data=edited_summary,
                file_name="professional_summary.txt",
                mime="text/plain",
                key="summary_download"
            )
        
        with col3:
            # Word count and character count
            word_count = len(edited_summary.split())
            char_count = len(edited_summary)
            st.markdown(f"**Word count:** {word_count} | **Character count:** {char_count}")

if __name__ == "__main__":
    main()
