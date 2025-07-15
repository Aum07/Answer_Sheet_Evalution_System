from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
import pandas as pd
import tempfile
import os
import time
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio
from functions import main, generate_pdf_report, generate_zip_reports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("answer-sheet-api")

# Configuration
class Settings:
    # Fixed paths to use current directory
    MODEL_PATH = "optimized_logistic_at_model.pkl"
    SCALER_PATH = "scaler.pkl"
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10 MB
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 10))
    API_KEY = os.getenv("API_KEY", "")  # Set to empty string to disable API key auth
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # 1 hour in seconds

settings = Settings()

# Define response models
class HealthResponse(BaseModel):
    status: str
    message: str
    model_file_exists: bool
    scaler_file_exists: bool
    version: str
    warning: Optional[str] = None

class EvaluationResult(BaseModel):
    Question: str
    Student_Answer: str
    Predicted_Score: int
    Feedback: Optional[str] = None

class StudentEvaluation(BaseModel):
    student_name: str
    results: List[EvaluationResult]
    total_score: int
    max_possible_score: int
    percentage: float

class MultipleEvaluationResponse(BaseModel):
    request_id: str
    evaluations: List[StudentEvaluation]
    timestamp: str

# Create cache for results
result_cache = {}

# Cache cleanup task
async def cleanup_cache():
    while True:
        current_time = time.time()
        keys_to_remove = []
        for key, (timestamp, _) in result_cache.items():
            if current_time - timestamp > settings.CACHE_TTL:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del result_cache[key]
        
        await asyncio.sleep(300)  # Check every 5 minutes

# Setup API
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Check if model files exist
    if not os.path.exists(settings.MODEL_PATH):
        logger.error(f"Model file not found at {settings.MODEL_PATH}")
    if not os.path.exists(settings.SCALER_PATH):
        logger.error(f"Scaler file not found at {settings.SCALER_PATH}")
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    # Start cache cleanup
    if settings.ENABLE_CACHE:
        cleanup_task = asyncio.create_task(cleanup_cache())
    
    yield
    
    # Cleanup on shutdown
    if settings.ENABLE_CACHE:
        cleanup_task.cancel()
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    logger.info("API shutting down, cleaned up temporary files")

app = FastAPI(
    title="Answer Sheet Evaluation API",
    description="API for evaluating student answer sheets against reference answers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    async def check_rate_limit(self, request: Request) -> bool:
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean up old requests
        self.requests = {ip: times for ip, times in self.requests.items()
                        if any(t > current_time - 60 for t in times)}
        
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        self.requests[client_ip].append(current_time)
        
        if len(self.requests[client_ip]) > self.requests_per_minute:
            return False
        return True

rate_limiter = RateLimiter(settings.RATE_LIMIT_PER_MINUTE)

async def check_rate_limit(request: Request):
    if not await rate_limiter.check_rate_limit(request):
        logger.warning(f"Rate limit exceeded for {request.client.host}")
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    return True

# API Key authentication (if enabled)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if settings.API_KEY and settings.API_KEY != "":
        if api_key != settings.API_KEY:
            logger.warning("Invalid API key used")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
    return True

# File handling utilities
@asynccontextmanager
async def save_upload_file_tmp(upload_file: UploadFile):
    try:
        suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="temp") as tmp:
            await upload_file.seek(0)
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = tmp.name
        await upload_file.seek(0)
        yield tmp_path
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

def validate_pdf(file_path: str) -> bool:
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception as e:
        logger.error(f"Error validating PDF: {str(e)}")
        return False

def check_file_size(file: UploadFile) -> bool:
    try:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        return size <= settings.MAX_UPLOAD_SIZE
    except Exception as e:
        logger.error(f"Error checking file size: {str(e)}")
        return False

# Extract student name from filename
def extract_student_name(filename: str) -> str:
    if not filename:
        return "Unknown Student"
    
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Replace underscores and hyphens with spaces
    name = name_without_ext.replace("_", " ").replace("-", " ")
    
    # Capitalize each word
    name = " ".join(word.capitalize() for word in name.split())
    
    return name

# Generate feedback based on score
def generate_feedback(score: int) -> str:
    if score >= 9:
        return "Excellent answer that demonstrates comprehensive understanding."
    elif score >= 7:
        return "Good answer with minor areas for improvement."
    elif score >= 5:
        return "Satisfactory answer but needs more detail or clarity."
    elif score >= 3:
        return "Answer needs significant improvement in content and structure."
    else:
        return "Answer is incomplete or shows limited understanding of the topic."

# Process a single student answer sheet
async def process_student_answer(reference_path: str, student_path: str, student_name: str, request_id: str):
    logger.info(f"{request_id}: Processing answer sheet for {student_name}")
    try:
        if not os.path.exists(settings.MODEL_PATH) or not os.path.exists(settings.SCALER_PATH):
            logger.error(f"Request {request_id}: Model files not found")
            return {
                "student_name": student_name,
                "error": "Model files not found"
            }

        result_df = main(reference_path, student_path, settings.MODEL_PATH, settings.SCALER_PATH)
        
        if not isinstance(result_df, pd.DataFrame) or 'Predicted Score' not in result_df.columns:
            logger.error(f"{request_id}: Invalid output from main() for {student_name}")
            return {
                "student_name": student_name,
                "error": "Model returned unexpected format"
            }

        # Add feedback
        result_df['Feedback'] = result_df['Predicted Score'].apply(generate_feedback)
        
        # Calculate total scores
        total_score = result_df['Predicted Score'].sum()
        max_possible_score = len(result_df) * 10
        percentage = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0

        # Format results
        simplified_df = result_df[['Question', 'Student Answer', 'Predicted Score', 'Feedback']]
        result_records = simplified_df.to_dict(orient='records')

        # Rename keys to match response model
        formatted_results = []
        for record in result_records:
            formatted_results.append({
                "Question": record["Question"],
                "Student_Answer": record["Student Answer"],
                "Predicted_Score": record["Predicted Score"],
                "Feedback": record["Feedback"]
            })

        return {
            "student_name": student_name,
            "results": formatted_results,
            "total_score": total_score,
            "max_possible_score": max_possible_score,
            "percentage": round(percentage, 2)
        }

    except Exception as e:
        logger.error(f"{request_id}: Error processing answer sheet for {student_name}: {str(e)}")
        return {
            "student_name": student_name,
            "error": f"Error processing answer sheet: {str(e)}"
        }

# Main endpoint for multiple student evaluations
@app.post("/evaluate/", response_model=MultipleEvaluationResponse, dependencies=[Depends(check_rate_limit), Depends(verify_api_key)])
async def evaluate_multiple_answer_sheets(
    background_tasks: BackgroundTasks,
    reference_pdf: UploadFile = File(...),
    student_pdfs: List[UploadFile] = File(...)
):
    request_id = f"req_{uuid.uuid4().hex}"
    logger.info(f"Request {request_id}: Processing multiple evaluation request for {len(student_pdfs)} students")

    # Validate reference PDF
    if not reference_pdf.filename.lower().endswith('.pdf'):
        logger.warning(f"Request {request_id}: Invalid reference file extension")
        raise HTTPException(status_code=400, detail="Reference file must be a PDF")

    if not check_file_size(reference_pdf):
        logger.warning(f"Request {request_id}: Reference file size exceeds limit")
        raise HTTPException(status_code=413, detail=f"Reference file exceeds the {settings.MAX_UPLOAD_SIZE/1024/1024}MB limit")

    # Validate student PDFs
    for student_pdf in student_pdfs:
        if not student_pdf.filename.lower().endswith('.pdf'):
            logger.warning(f"Request {request_id}: Invalid student file extension: {student_pdf.filename}")
            raise HTTPException(status_code=400, detail=f"Student file {student_pdf.filename} must be a PDF")

        if not check_file_size(student_pdf):
            logger.warning(f"Request {request_id}: Student file size exceeds limit: {student_pdf.filename}")
            raise HTTPException(status_code=413, detail=f"Student file {student_pdf.filename} exceeds the {settings.MAX_UPLOAD_SIZE/1024/1024}MB limit")

    try:
        # Save reference PDF to temp file
        async with save_upload_file_tmp(reference_pdf) as ref_path:
            if not validate_pdf(ref_path):
                logger.warning(f"Request {request_id}: Invalid reference PDF content")
                raise HTTPException(status_code=400, detail="Invalid reference PDF file")

            # Process each student PDF
            evaluations = []
            for student_pdf in student_pdfs:
                student_name = extract_student_name(student_pdf.filename)
                
                async with save_upload_file_tmp(student_pdf) as student_path:
                    if not validate_pdf(student_path):
                        logger.warning(f"Request {request_id}: Invalid student PDF content: {student_pdf.filename}")
                        evaluations.append({
                            "student_name": student_name,
                            "error": "Invalid PDF file"
                        })
                        continue

                    result = await process_student_answer(ref_path, student_path, student_name, request_id)
                    
                    if "error" in result:
                        evaluations.append({
                            "student_name": student_name,
                            "error": result["error"]
                        })
                    else:
                        evaluations.append(result)

            # Filter out errors
            successful_evaluations = [eval for eval in evaluations if "error" not in eval]
            
            if not successful_evaluations:
                raise HTTPException(status_code=500, detail="All student evaluations failed")

            response = {
                "request_id": request_id,
                "evaluations": successful_evaluations,
                "timestamp": datetime.now().isoformat()
            }

            # Cache the results for download functionality
            result_cache[request_id] = (time.time(), response)

            logger.info(f"Request {request_id}: Successfully processed {len(successful_evaluations)} student answer sheets")
            return response

    except Exception as e:
        logger.error(f"Request {request_id}: Error processing multiple PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")

# Download individual student report
@app.get("/download/{request_id}")
async def download_evaluation(request_id: str, student: Optional[str] = None):
    """Download individual student evaluation report as PDF"""
    
    # Check if the evaluation exists in cache
    if request_id not in result_cache:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    _, response_data = result_cache[request_id]
    evaluations = response_data.get("evaluations", [])
    
    if student:
        # Find specific student
        student_data = None
        for eval_data in evaluations:
            if eval_data["student_name"].lower() == student.lower():
                student_data = eval_data
                break
        
        if not student_data:
            raise HTTPException(status_code=404, detail=f"Student '{student}' not found in evaluation")
        
        # Generate PDF report for single student
        pdf_path = generate_pdf_report(student_data, request_id)
        safe_name = student_data["student_name"].replace(" ", "_")
        filename = f"{safe_name}_evaluation_report.pdf"
        
    else:
        # If no specific student, return first student's report
        if not evaluations:
            raise HTTPException(status_code=404, detail="No evaluations found")
        
        student_data = evaluations[0]
        pdf_path = generate_pdf_report(student_data, request_id)
        safe_name = student_data["student_name"].replace(" ", "_")
        filename = f"{safe_name}_evaluation_report.pdf"

    return FileResponse(
        path=pdf_path,
        filename=filename,
        media_type="application/pdf",
        background=BackgroundTasks().add_task(lambda: os.unlink(pdf_path) if os.path.exists(pdf_path) else None)
    )

# Download multiple evaluations as ZIP
@app.get("/download-multiple/{request_id}")
async def download_multiple_evaluations(request_id: str):
    """Download all student evaluation reports as a ZIP file"""
    
    # Check if the evaluation exists in cache
    if request_id not in result_cache:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    _, response_data = result_cache[request_id]
    evaluations = response_data.get("evaluations", [])
    
    if not evaluations:
        raise HTTPException(status_code=404, detail="No evaluations found")
    
    # Generate ZIP file with all reports
    zip_path = generate_zip_reports(evaluations, request_id)
    
    return FileResponse(
        path=zip_path,
        filename=f"evaluation_reports_{request_id}.zip",
        media_type="application/zip",
        background=BackgroundTasks().add_task(lambda: os.unlink(zip_path) if os.path.exists(zip_path) else None)
    )

@app.get("/health/", response_model=HealthResponse)
async def health_check():
    model_exists = os.path.exists(settings.MODEL_PATH)
    scaler_exists = os.path.exists(settings.SCALER_PATH)
    
    status = {
        "status": "healthy" if model_exists and scaler_exists else "degraded",
        "message": "Answer Sheet Evaluation API is operational",
        "model_file_exists": model_exists,
        "scaler_file_exists": scaler_exists,
        "version": "1.0.0"
    }

    if not model_exists or not scaler_exists:
        status["warning"] = "Model files missing. API may not function correctly."

    return status

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request to {request.url.path} processed in {process_time:.4f} seconds")
    return response

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    logger.warning("Static files directory not found, skipping static file serving")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Answer Sheet Evaluation API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
