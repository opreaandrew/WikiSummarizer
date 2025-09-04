import os
import time
import json
import requests
import logging
from pathlib import Path
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="WikiSummarizer - Web UI")
templates = Jinja2Templates(directory="templates")

# Configure logging to filter out health check endpoints
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return not (record.getMessage().find("GET /health") >= 0)

# Apply filter to uvicorn access logger
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

# Configure basic service logging
logging.basicConfig(
    level=logging.INFO,
    format="[web-ui] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("web-ui")

class PipelineOrchestrator:
    def __init__(self, data_dir="/app/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def trigger_scraping(self, topic: str):
        """Trigger the scraping service"""
        request_file = self.data_dir / "scrape_request.txt"
        with open(request_file, 'w') as f:
            f.write(topic)
        logger.info(f"Scraping request sent for topic: {topic}")
    
    def wait_for_training_completion(self, timeout=300):
        """Wait for the entire pipeline to complete"""
        train_complete_file = self.data_dir / "train_complete.txt"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if train_complete_file.exists():
                with open(train_complete_file, 'r') as f:
                    content = f.read().strip()
                    topic, model_path = content.split('|', 1)
                train_complete_file.unlink()
                return topic, model_path
            time.sleep(2)
        
        return None, None
    
    def get_scraped_content(self, topic: str):
        """Get the scraped content for generating a summary"""
        scraped_file = self.data_dir / f"{topic.replace(' ', '_')}_scraped.json"
        
        if scraped_file.exists():
            with open(scraped_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Combine all article content
            combined_content = ""
            for article in data['articles']:
                combined_content += f"{article['title']}: {article['summary']} "
            
            return combined_content.strip()
        
        return None

# Initialize orchestrator
orchestrator = PipelineOrchestrator()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_topic(request: Request, topic: str = Form(...)):
    """Process a topic through the entire pipeline"""
    try:
        logger.info(f"Processing topic: {topic}")
        
        # Step 1: Trigger scraping
        orchestrator.trigger_scraping(topic)
        
        # Step 2: Wait for pipeline completion
        result_topic, model_path = orchestrator.wait_for_training_completion()
        
        if not result_topic:
            return templates.TemplateResponse(
                "result.html", 
                {
                    "request": request, 
                    "topic": topic, 
                    "error": "Pipeline timeout or error occurred"
                }
            )
        
        # Step 3: Get scraped content for summarization
        content = orchestrator.get_scraped_content(topic)
        
        if not content:
            return templates.TemplateResponse(
                "result.html", 
                {
                    "request": request, 
                    "topic": topic, 
                    "error": "No content available for summarization"
                }
            )
        
        # Step 4: Generate summary
        try:
            summary_response = requests.post(
                "http://summarizer:8001/summarize",
                json={"topic": topic.replace(' ', '_'), "text": content},
                timeout=120  # Increased timeout to 120 seconds
            )
            
            if summary_response.status_code == 200:
                summary_data = summary_response.json()
                summary = summary_data["summary"]
            else:
                summary = "Error generating summary from the trained model"
                
        except Exception as e:
            logger.error(f"Error calling summarizer service: {e}")
            summary = f"Error connecting to summarizer service: {str(e)}"
        
        return templates.TemplateResponse(
            "result.html", 
            {
                "request": request, 
                "topic": topic, 
                "summary": summary,
                "model_path": str(model_path) if model_path else "fallback model"
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing topic: {e}")
        return templates.TemplateResponse(
            "result.html", 
            {
                "request": request, 
                "topic": topic, 
                "error": f"Pipeline error: {str(e)}"
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "web-ui"}

@app.get("/models")
async def available_models():
    """Get available models from summarizer service"""
    try:
        response = requests.get("http://summarizer:8001/models", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"available_models": [], "error": "Could not fetch models"}
    except Exception as e:
        return {"available_models": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
