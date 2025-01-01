from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

from src.models.generic_gpt2_model import GenericGPT2Model
from src.parsers.ats_parser import ATSParser
from src.parsers.industry_manager_parser import IndustryManagerParser

# Prometheus metrics
REQUESTS_TOTAL = Counter(
    'resume_video_requests_total',
    'Total number of resume video script requests',
    ['template_type']
)

PROCESSING_TIME = Histogram(
    'resume_video_processing_seconds',
    'Time spent processing resume video script requests',
    ['template_type']
)

ERROR_COUNT = Counter(
    'resume_video_errors_total',
    'Total number of errors in resume video script generation',
    ['template_type', 'error_type']
)

app = FastAPI(
    title="Resume Video Script Generator API",
    description="API for generating video scripts from resume templates",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
gpt2_model = GenericGPT2Model()

class ScriptResponse(BaseModel):
    script: str
    template_type: str

@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate-script", response_model=ScriptResponse)
async def generate_script(
    file: UploadFile = File(...),
    template_type: str = Form(...),
):
    start_time = time.time()
    temp_path = None
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Use parser based on user's template selection
        if template_type.lower() == "ats":
            parser = ATSParser(temp_path)
            template_label = "ATS/HR"
        elif template_type.lower() == "industry":
            parser = IndustryManagerParser(temp_path)
            template_label = "Industry Manager"
        else:
            ERROR_COUNT.labels(template_type="unknown", error_type="invalid_template").inc()
            raise HTTPException(
                status_code=400,
                detail="Invalid template type. Must be either 'ats' or 'industry'"
            )
        
        # Parse resume and generate script
        resume_data = parser.parse()
        script = gpt2_model.generate_summary(resume_data)
        
        # Record metrics
        REQUESTS_TOTAL.labels(template_type=template_label).inc()
        PROCESSING_TIME.labels(template_type=template_label).observe(time.time() - start_time)
        
        return ScriptResponse(script=script, template_type=template_label)
    
    except Exception as e:
        # Record error metrics
        error_type = type(e).__name__
        ERROR_COUNT.labels(
            template_type=template_type.lower(),
            error_type=error_type
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
