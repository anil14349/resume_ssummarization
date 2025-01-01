import streamlit as st
import requests
import os
from tempfile import NamedTemporaryFile

# Configure page
st.set_page_config(
    page_title="Resume Video Script Generator",
    page_icon="🎥",
    layout="wide"
)

def main():
    st.title("Resume Video Script Generator 🎥")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application generates video scripts from resume templates. "
        "Select your template type and upload your resume to get started."
    )
    
    # Main content
    st.header("Generate Video Script")
    
    # Template selection
    template_type = st.radio(
        "Select Resume Template Type",
        options=["ATS/HR Resume", "Industry Manager Resume"],
        help="Choose the type of resume template you're using"
    )
    
    # Convert selection to API parameter
    template_param = "ats" if template_type == "ATS/HR Resume" else "industry"
    
    # File uploader
    uploaded_file = st.file_uploader(
        f"Upload your {template_type}",
        type=["docx"],
        help="Upload your resume in .docx format"
    )
    
    if uploaded_file:
        st.success("File uploaded successfully!")
        
        # Display file info
        st.write("File details:")
        st.json({
            "Filename": uploaded_file.name,
            "Size": f"{uploaded_file.size / 1024:.2f} KB",
            "Type": uploaded_file.type,
            "Template": template_type
        })
        
        # Generate button
        if st.button("Generate Video Script", type="primary"):
            with st.spinner("Generating video script..."):
                try:
                    # Create API request with template type
                    files = {"file": uploaded_file}
                    data = {"template_type": template_param}
                    
                    response = requests.post(
                        "http://localhost:8000/generate-script",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display results
                        st.header("Generated Script")
                        st.info(f"Template Type: {data['template_type']}")
                        
                        # Display script in a text area
                        st.text_area(
                            "Generated Script",
                            value=data['script'],
                            height=300,
                            disabled=True
                        )
                        
                        # Add download button
                        st.download_button(
                            label="Download Script",
                            data=data['script'],
                            file_name=f"{template_type.lower().replace('/', '_')}_script.txt",
                            mime="text/plain"
                        )
                        
                        # Add helpful tips based on template type
                        if template_type == "ATS/HR Resume":
                            st.info(
                                "💡 Tip: This script is optimized for HR and recruitment "
                                "positions, highlighting skills and achievements relevant "
                                "to human resources."
                            )
                        else:
                            st.info(
                                "💡 Tip: This script is tailored for industry management "
                                "roles, emphasizing leadership experience and strategic "
                                "accomplishments."
                            )
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                
                except Exception as e:
                    st.error(f"Error connecting to the API: {str(e)}")

if __name__ == "__main__":
    main()
