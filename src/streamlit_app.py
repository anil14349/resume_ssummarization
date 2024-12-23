import streamlit as st
import os
from pathlib import Path
import tempfile
import logging
import sys

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from generate_summary import main as generate_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_css():
    """Load custom CSS styles."""
    st.markdown("""
        <style>
        /* Modern color scheme */
        :root {
            --primary: #2563eb;
            --primary-light: #3b82f6;
            --primary-dark: #1d4ed8;
            --secondary: #64748b;
            --success: #059669;
            --success-light: #d1fae5;
            --warning: #d97706;
            --error: #dc2626;
            --background: #f8fafc;
            --surface: #ffffff;
            --surface-hover: #f1f5f9;
            --border: #e2e8f0;
            --text: #0f172a;
            --text-light: #475569;
            --shadow: rgba(0, 0, 0, 0.1);
        }
        
        /* Global styles */
        .stApp {
            background-color: var(--background);
            color: var(--text);
        }
        
        .stApp > header {
            background-color: transparent;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            font-weight: 600;
            letter-spacing: -0.025em;
        }
        
        p, span, div {
            color: var(--text);
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
            border-radius: 16px;
            margin: 1rem 0 3rem 0;
            box-shadow: 0 4px 6px -1px var(--shadow), 0 2px 4px -2px var(--shadow);
        }
        
        .main-header h1 {
            color: white !important;
            font-size: 2.75rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .main-header p {
            color: rgba(255, 255, 255, 0.95) !important;
            font-size: 1.25rem;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }
        
        /* Section headers */
        h2 {
            font-size: 1.5rem;
            margin: 1.5rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary-light);
            color: var(--primary-dark);
        }
        
        /* Card styling */
        .card {
            background: var(--surface);
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px 0 var(--shadow), 0 1px 2px -1px var(--shadow);
            margin: 1rem 0;
            border: 1px solid var(--border);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px var(--shadow), 0 2px 4px -2px var(--shadow);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 0.025em;
            transition: all 0.2s;
            box-shadow: 0 1px 3px 0 var(--shadow);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px var(--shadow);
            background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%);
        }
        
        /* Summary box styling */
        .summary-box {
            background-color: var(--surface);
            border-radius: 12px;
            padding: 2rem;
            margin: 1.5rem 0;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px 0 var(--shadow);
        }
        
        .stTextArea textarea {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-size: 1rem;
            line-height: 1.7;
            color: var(--text);
            background-color: var(--surface);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            min-height: 200px;
            box-shadow: inset 0 1px 2px var(--shadow);
        }
        
        .stTextArea textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        /* Success message styling */
        .success-message {
            background-color: var(--success-light);
            color: var(--success);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border: 1px solid rgba(5, 150, 105, 0.2);
        }
        
        /* File uploader styling */
        .uploadedFile {
            border: 2px dashed var(--primary-light);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            background: var(--surface);
            transition: all 0.2s;
        }
        
        .uploadedFile:hover {
            border-color: var(--primary);
            background: var(--surface-hover);
        }
        
        /* Radio and select styling */
        .stRadio > label, .stSelectbox > label {
            color: var(--text);
            font-weight: 500;
            margin-bottom: 0.75rem;
            font-size: 1rem;
        }
        
        .stRadio > div {
            background: var(--surface);
            padding: 0.5rem;
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        
        .stSelectbox > div > div {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        
        /* Metrics styling */
        .metric-container {
            background: var(--surface);
            padding: 1.25rem;
            border-radius: 10px;
            border: 1px solid var(--border);
            margin: 0.75rem 0;
            text-align: center;
            transition: transform 0.2s;
        }
        
        .metric-container:hover {
            transform: translateY(-2px);
        }
        
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.25rem;
        }
        
        .metric-label {
            color: var(--text-light);
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        /* Custom container styling */
        .content-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        /* Spinner styling */
        .stSpinner > div {
            border-color: var(--primary-light);
            border-right-color: transparent;
        }
        
        /* Code block styling */
        .stCodeBlock {
            background: var(--surface);
            border-radius: 8px;
            border: 1px solid var(--border);
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def main():
    # Page config
    st.set_page_config(
        page_title="Resume Summary Generator",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_css()
    
    # Main header
    st.markdown("""
        <div class="main-header">
            <h1>📄 Resume Summary Generator</h1>
            <p>Transform your resume into a professional summary using advanced language models</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for main content
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown('<h2>📤 Upload Resume</h2>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload your resume (DOCX format)",
                type=["docx"],
                help="Drag and drop your resume file here"
            )
            
            if uploaded_file:
                st.markdown('<div class="success-message">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
                st.markdown(f"**File name:** {uploaded_file.name}")
                st.markdown(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2>⚙️ Settings</h2>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Template/Parser selection
            template_type = st.radio(
                "Select Resume Type:",
                ["ATS Classic HR Resume", "Industry Manager Resume"],
                help="Choose the type of resume you're using"
            )
            
            # Set parser type based on template
            parser_type = "ats" if template_type == "ATS Classic HR Resume" else "industry"
            
            # Model selection
            model_type = st.selectbox(
                "Select AI Model:",
                ["gpt2", "t5", "bart"],
                help="Choose the model for summary generation"
            )
            
            # Debug mode
            debug = st.checkbox("Debug Mode", help="Show detailed processing information")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate summary button (centered)
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    generate_button = st.button("🚀 Generate Summary", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate summary
    if uploaded_file and generate_button:
        with st.spinner("🔄 Processing your resume..."):
            try:
                # Save uploaded file
                temp_file_path = save_uploaded_file(uploaded_file)
                if not temp_file_path:
                    st.error("Failed to process uploaded file")
                    return
                
                try:
                    # Generate summary using the main function from generate_summary.py
                    summary = generate_summary(
                        input_file=temp_file_path,
                        model_type=model_type,
                        parser_type=parser_type,
                        debug=debug
                    )
                    
                    # Display results
                    st.markdown('<div class="success-message">✅ Summary generated successfully!</div>', unsafe_allow_html=True)
                    
                    # Display summary in a nice format with editing capability
                    st.markdown('<h3>📝 Generated Summary</h3>', unsafe_allow_html=True)
                    
                    # Editable text area
                    edited_summary = st.text_area(
                        "Edit your summary:",
                        value=summary,
                        height=300,
                        key="summary_editor",
                        help="You can edit the generated summary here"
                    )
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📋 Copy to Clipboard"):
                            st.code(edited_summary)
                            st.success("Copied to clipboard!")
                    
                    with col2:
                        st.download_button(
                            "📥 Download Summary",
                            edited_summary,
                            file_name=f"{Path(uploaded_file.name).stem}_summary.txt",
                            mime="text/plain"
                        )
                    
                    with col3:
                        st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{len(edited_summary.split())} words</div>
                                <div class="metric-label">Summary Length</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    if debug:
                        st.exception(e)
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                if debug:
                    st.exception(e)

if __name__ == "__main__":
    main()
