import streamlit as st
import os
from pathlib import Path
from generate_summary import main as generate_summary
from tempfile import NamedTemporaryFile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Resume Summary Generator",
    page_icon="📄",
    layout="wide"
)

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return the path."""
    try:
        suffix = Path(uploaded_file.name).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def main():
    # Title and description
    st.title("📄 Resume Summary Generator")
    st.markdown("""
    Upload your resume (DOCX format) and get a professional summary generated using advanced language models.
    """)
    
    # Sidebar for options
    with st.sidebar:
        st.header("⚙️ Options")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["gpt2", "t5", "bart"],
            help="Choose the language model for summary generation"
        )
        
        # Parser selection
        parser_type = st.selectbox(
            "Select Parser",
            ["ats", "industry"],
            help="ATS parser is optimized for standard resumes, Industry parser for specific domains"
        )
        
        # Debug mode
        debug = st.checkbox("Debug Mode", help="Enable detailed logging")
        
        st.markdown("---")
        st.markdown("""
        ### 📋 Supported Format
        - Word Document (.docx)
        """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["docx"],
        help="Drag and drop your resume file here (DOCX format only)"
    )

    if uploaded_file:
        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"File: {uploaded_file.name}")
        with col2:
            st.info(f"Size: {uploaded_file.size / 1024:.2f} KB")

        # Generate button
        if st.button("🚀 Generate Summary", type="primary"):
            with st.spinner("Generating summary..."):
                try:
                    # Save uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        try:
                            # Generate summary
                            summary = generate_summary(
                                input_file=file_path,
                                model_type=model_type,
                                parser_type=parser_type,
                                debug=debug
                            )
                            
                            # Display results
                            st.success("✅ Summary generated successfully!")
                            
                            # Display summary in an editable text area
                            st.markdown("### 📝 Generated Summary")
                            st.markdown("""
                            <style>
                            .stTextArea textarea {
                                font-size: 1rem;
                                color: #0f1116;
                                background-color: #f0f2f6;
                                min-height: 200px;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            edited_summary = st.text_area(
                                "Edit your summary",
                                value=summary,
                                height=300,
                                key="summary_editor"
                            )
                            
                            # Add download button for edited summary
                            st.download_button(
                                "📥 Download Summary",
                                edited_summary,
                                file_name=f"{Path(uploaded_file.name).stem}_summary.txt",
                                mime="text/plain"
                            )
                            
                        finally:
                            # Clean up temporary file
                            if os.path.exists(file_path):
                                os.unlink(file_path)
                                
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    logger.error(f"Error processing file: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center">
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
