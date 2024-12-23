from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import os
from typing import Optional
from tempfile import NamedTemporaryFile

from generate_summary import main as generate_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Resume Summary Generator API",
    description="API for generating professional summaries from resumes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/generate-summary/")
async def generate_resume_summary(
    file: UploadFile = File(...),
    model_type: str = "gpt2",
    parser_type: str = "ats",
    debug: bool = False
):
    """
    Generate a professional summary from a resume file.
    
    Args:
        file: Resume file (DOCX only)
        model_type: Model to use (gpt2, t5, or bart)
        parser_type: Parser to use (ats or industry)
        debug: Enable debug logging
        
    Returns:
        JSON response with generated summary
    """
    try:
        # Validate file type
        allowed_extensions = {".docx"}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Only DOCX files are supported."
            )
            
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
        try:
            # Generate summary
            summary = generate_summary(
                input_file=temp_file_path,
                model_type=model_type,
                parser_type=parser_type,
                debug=debug
            )
            
            return JSONResponse(content={
                "filename": file.filename,
                "model_type": model_type,
                "parser_type": parser_type,
                "summary": summary
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}"
        )

@app.get("/models")
async def list_models():
    """List available models for summary generation."""
    return {
        "models": [
            {
                "id": "gpt2",
                "name": "GPT-2",
                "description": "OpenAI's GPT-2 model fine-tuned for resume summarization"
            },
            {
                "id": "t5",
                "name": "T5",
                "description": "Google's T5 model fine-tuned for resume summarization"
            },
            {
                "id": "bart",
                "name": "BART",
                "description": "Facebook's BART model fine-tuned for resume summarization"
            }
        ]
    }

@app.get("/parsers")
async def list_parsers():
    """List available resume parsers."""
    return {
        "parsers": [
            {
                "id": "ats",
                "name": "ATS Parser",
                "description": "Parser optimized for ATS (Applicant Tracking System) formatted resumes"
            },
            {
                "id": "industry",
                "name": "Industry Parser",
                "description": "Parser optimized for industry-specific resume formats"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
