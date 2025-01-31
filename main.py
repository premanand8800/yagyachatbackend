from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.utils.input_validator import InputValidator
from app.models.user_input import UserInput
from datetime import datetime
import logging
import traceback
import uvicorn
import sys
import os

# Ensure the app directory is in the sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(title="Rishi API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputRequest(BaseModel):
    input_text: str
    previous_input: str | None = None

# Initialize the input validator
validator = InputValidator()

@app.post("/api/process_input")
async def process_input(request: InputRequest):
    """
    Process and validate user input using the InputValidator
    """
    try:
        # Log the incoming request
        logging.info(f"Received request with input: {request.input_text}")
        
        # Create UserInput object with metadata
        user_input = UserInput(
            raw_input=request.input_text,
            metadata={
                "previous_input": request.previous_input,
                "timestamp": datetime.now().isoformat()
            } if request.previous_input else {
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Validate the input
        validation_result = await validator.validate_input(user_input)
        
        # Create detailed response
        result = {
            "status": "error" if not validation_result.is_valid else "success",
            "input": {
                "text": request.input_text,
                "length": len(request.input_text),
                "previous_input": request.previous_input
            },
            "validation": {
                "is_valid": validation_result.is_valid,
                "input_type": validation_result.input_type,
                "content_analysis": {
                    "has_background": validation_result.has_background,
                    "background_completeness": round(validation_result.background_completeness, 2),
                    "has_goals": validation_result.has_goals,
                    "goals_clarity": round(validation_result.goals_clarity, 2),
                    "overall_clarity": round(validation_result.clarity_score, 2),
                    "context_score": round(validation_result.context_score, 2)
                },
                "safety": {
                    "safety_score": round(validation_result.safety_score, 2),
                    "error_message": validation_result.error_message
                },
                "suggestions": validation_result.suggestions if hasattr(validation_result, 'suggestions') else [],
                "clarification_questions": validation_result.clarification_questions if hasattr(validation_result, 'clarification_questions') else []
            }
        }
        
        # Return 400 for invalid input, 200 for valid input
        status_code = 400 if not validation_result.is_valid else 200
        return JSONResponse(status_code=status_code, content=result)
        
    except Exception as e:
        error_msg = f"Error processing input: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Rishi API is running"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8005, reload=True)
