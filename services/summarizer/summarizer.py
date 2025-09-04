from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging

app = FastAPI(title="WikiSummarizer - Summarizer Service")

# Configure logging to filter out health check endpoints
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return not (record.getMessage().find("GET /health") >= 0)

# Apply filter to uvicorn access logger
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

# Configure basic service logging
logging.basicConfig(
    level=logging.INFO,
    format="[summarizer] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("summarizer")

class SummarizationRequest(BaseModel):
    topic: str
    text: str

class SummarizationResponse(BaseModel):
    summary: str
    topic: str

class WikiSummarizer:
    def __init__(self, model_dir="/app/models"):
        self.model_dir = Path(model_dir)
        self.loaded_models = {}
        self.fallback_model_name = "facebook/bart-base"
        
    def load_model(self, topic: str):
        """Load the fine-tuned model for a specific topic"""
        if topic in self.loaded_models:
            return self.loaded_models[topic]
        
        model_path = self.model_dir / f"{topic}_model"
        
        try:
            if model_path.exists():
                logger.info(f"Loading fine-tuned model for topic: {topic}")
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
            else:
                logger.info(f"Fine-tuned model not found for {topic}, using fallback model")
                tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.fallback_model_name)
            
            self.loaded_models[topic] = (tokenizer, model)
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"Error loading model for {topic}: {e}")
            logger.info("Using fallback model")
            tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.fallback_model_name)
            self.loaded_models[topic] = (tokenizer, model)
            return tokenizer, model
    
    def generate_summary(self, topic: str, text: str, max_length: int = 150) -> str:
        """Generate a summary using the appropriate model"""
        try:
            tokenizer, model = self.load_model(topic)
            
            # Tokenize input
            inputs = tokenizer.encode(
                text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Generate summary
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=30,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode summary
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

# Initialize summarizer
summarizer = WikiSummarizer()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "summarizer"}

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """Generate a summary for the given text and topic"""
    try:
        summary = summarizer.generate_summary(request.topic, request.text)
        return SummarizationResponse(summary=summary, topic=request.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_available_models():
    """List available fine-tuned models"""
    model_dir = Path("/app/models")
    available_models = []
    
    if model_dir.exists():
        for model_path in model_dir.iterdir():
            if model_path.is_dir() and model_path.name.endswith("_model"):
                topic = model_path.name.replace("_model", "")
                available_models.append(topic)
    
    return {"available_models": available_models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
