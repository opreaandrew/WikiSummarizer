# WikiSummarizer – Modular NLP Pipeline with Containers

## Overview

WikiSummarizer is a **learning-focused, modular, and containerized project** that demonstrates how to build and deploy an **end-to-end NLP pipeline** using only **free and open data sources (Wikipedia)** and **open-source tools**.  

Through a simple web interface, users can input a topic, automatically scrape related Wikipedia articles, preprocess the text, fine-tune a transformer model, and generate **summarized outputs**.  

This project emphasizes **accessibility and transparency**, making advanced AI concepts practical and approachable.  
It showcases:  
- End-to-end ML pipelines in a modular, containerized architecture  
- Real-world NLP applications such as text summarization  
- Practical DevOps workflows (Docker, orchestration, modular services)  
- Open knowledge: free data from Wikipedia and free tools (e.g., Hugging Face)  

The goal is to provide a **hands-on learning journey** into how AI/ML systems are designed and deployed in production, while keeping the system lightweight, extensible, and easy to experiment with.

## Quick Start

1. **Start the pipeline:**
   ```bash
   docker compose up --build
   ```

2. **Access the web interface:**  
   Open your browser to: http://localhost:8000

3. **Generate a summary:**  
   Enter a topic (e.g., "Artificial Intelligence") and click "Start Pipeline"

4. **Wait and enjoy:**  
   The pipeline takes 5-10 minutes on first run, then displays your AI-generated summary!  


---

## Architecture

The pipeline consists of **5 containerized services**, each isolated and lightweight with minimal dependencies. Services communicate via shared Docker volumes and a bridge network.

### Service Details

1. **🔍 Scraper Service**
   - **Input**: Topic string from the web UI
   - **Function**: Scrapes relevant Wikipedia content using the Wikipedia API
   - **Output**: Raw JSON files with article content stored in shared volume
   - **Dependencies**: `wikipedia`, `requests`, `beautifulsoup4`

2. **🧹 Preprocessor Service**
   - **Input**: Raw scraped text from shared volume
   - **Function**: Cleans, tokenizes, and normalizes data for model training
   - **Output**: Preprocessed JSONL training dataset
   - **Dependencies**: `nltk`, `transformers`

3. **🤖 Trainer Service**
   - **Input**: Preprocessed dataset
   - **Function**: Fine-tunes a BART transformer model on topic-specific data
   - **Output**: Trained model checkpoint saved to model volume
   - **Dependencies**: `torch`, `transformers`, `datasets`, `accelerate`, `numpy<2.0`, `pyarrow<15.0.0`

4. **✨ Summarizer Service**
   - **Input**: Text to summarize + topic context
   - **Function**: Generates summaries using fine-tuned or fallback models
   - **Output**: AI-generated summary via FastAPI
   - **Dependencies**: `torch`, `transformers`, `fastapi`, `uvicorn`, `numpy<2.0`

5. **🌐 Web UI Service**
   - **Input**: User topic via web form
   - **Function**: Orchestrates the entire pipeline and provides user interface
   - **Output**: Displays results and coordinates service communication
   - **Dependencies**: `fastapi`, `jinja2`, `requests`

---

## Data Flow & Communication

```
User Input → Web UI → Scraper → Preprocessor → Trainer → Summarizer → Web UI → User
```

**Detailed Pipeline Steps:**
1. **User enters topic** in web interface (e.g., "Artificial Intelligence")
2. **Web UI triggers Scraper** via shared file coordination
3. **Scraper** fetches Wikipedia articles and saves raw data to `/app/data/`
4. **Preprocessor** automatically processes scraped data and creates training pairs
5. **Trainer** fine-tunes BART model and saves checkpoint to `/app/models/`
6. **Summarizer** loads trained model and generates summary via API call
7. **Web UI** displays AI-generated summary to user

**Container Communication:**
- **Shared Volumes**: `data_volume` (scraped/processed data) and `model_volume` (model checkpoints)
- **Bridge Network**: All services communicate using service names (e.g., `http://summarizer:8001`)
- **File Coordination**: Services coordinate through marker files in shared volumes
- **Health Checks**: Services implement health checks to ensure orderly startup

## Technologies Used

- **Python 3.11** for all NLP services
- **Wikipedia API** for content scraping  
- **Hugging Face Transformers** (BART model) for fine-tuning and inference
- **NLTK** for text preprocessing and tokenization
- **FastAPI + Uvicorn** for web services and APIs
- **Docker & Docker Compose** for containerization and orchestration
- **Shared Volumes & Bridge Networks** for inter-service communication

## Project Structure

```
WikiSummarizer/
├── docker-compose.yml          # Orchestrates all 5 services with health checks
├── README.md                   # This file
└── services/                   # Individual containerized services
    ├── scraper/               # Wikipedia scraping service
    │   ├── Dockerfile
    │   ├── requirements.txt   # wikipedia, requests, beautifulsoup4
    │   └── scraper.py
    ├── preprocessor/          # Text preprocessing service  
    │   ├── Dockerfile
    │   ├── requirements.txt   # nltk, transformers
    │   ├── start.sh          # Service startup script
    │   └── preprocessor.py
    ├── trainer/               # Model fine-tuning service
    │   ├── Dockerfile
    │   ├── requirements.txt   # torch, transformers, datasets, numpy<2.0, pyarrow<15.0.0
    │   ├── start.sh          # Service startup script
    │   └── trainer.py
    ├── summarizer/            # Summarization API service
    │   ├── Dockerfile
    │   ├── requirements.txt   # torch, transformers, fastapi, numpy<2.0
    │   └── summarizer.py
    └── web-ui/                # Web interface service
        ├── Dockerfile
        ├── requirements.txt   # fastapi, jinja2, requests
        ├── main.py
        └── templates/
            ├── index.html     # Main interface
            └── result.html    # Results display
```

## Usage Examples

### Basic Usage
```bash
# Start the complete pipeline
docker compose up --build

# Open web interface
# Navigate to: http://localhost:8000

# Enter topics like:
# - "Artificial Intelligence"
# - "Quantum Computing" 
# - "Climate Change"
# - "Blockchain Technology"
```

### API Testing
```bash
# Test summarizer service directly
curl -X POST "http://localhost:8001/summarize" \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI", "text": "Your text to summarize here..."}'

# Check available models
curl http://localhost:8001/models

# Health checks
curl http://localhost:8000/health  # Web UI
curl http://localhost:8001/health  # Summarizer
```

### Development Commands
```bash
# View logs from all services
docker compose logs

# View logs from specific service
docker compose logs scraper
docker compose logs trainer

# Stop and cleanup
docker compose down -v

# Restart services
docker compose restart
```

## Troubleshooting

### Common Issues

**Pipeline Timeout**
- First run can take 5-10 minutes (model downloads + training)
- Try more specific topics for faster processing
- Check logs: `docker compose logs trainer`

**NumPy/PyArrow Compatibility**
- System now uses compatible versions: numpy<2.0, pyarrow<15.0.0
- This resolves common transformer library errors

**Memory Issues**
- Training requires ~2GB RAM minimum
- Close other applications during first run
- Monitor with: `docker stats`

**No Wikipedia Results**
- Verify topic exists on Wikipedia
- Try different phrasing or broader topics
- Check scraper logs: `docker compose logs scraper`

**Model Download Issues**
- First run downloads BART model (~500MB)
- Subsequent runs use cached models
- Ensure stable internet connection

### Debugging
```bash
# View real-time logs
docker compose logs -f

# Check service status  
docker compose ps

# Access container shell for debugging
docker compose exec web-ui /bin/bash
docker compose exec trainer /bin/bash
```

## Customization & Extension

### Modify the Base Model
Edit the model name in `services/trainer/trainer.py` and `services/summarizer/summarizer.py`:
```python
self.model_name = "facebook/bart-large"  # or "t5-base", "distilgpt2", etc.
```

### Adjust Training Parameters
Modify training settings in `services/trainer/trainer.py`:
```python
training_args = TrainingArguments(
    num_train_epochs=3,          # More epochs for better quality
    per_device_train_batch_size=4, # Larger batches (requires more RAM)
    learning_rate=3e-5,          # Experiment with learning rates
)
```

### Add New Data Sources
Extend `services/scraper/scraper.py` to include:
- PDF documents  
- News articles
- Custom text files
- RSS feeds

### Scale for Production
- Add multiple summarizer instances for load balancing
- Implement caching with Redis for repeated topics  
- Add persistent storage: map volumes to host directories
- Include monitoring with Prometheus/Grafana
- Add authentication and rate limiting

## Educational Value

This project demonstrates **real-world ML engineering concepts**:

- **Microservices Architecture**: Independent, scalable services
- **Container Orchestration**: Docker Compose coordination
- **ML Pipeline Design**: Data flow from raw input to model output  
- **API Development**: RESTful services with FastAPI
- **Model Training**: Fine-tuning transformers with Hugging Face
- **DevOps Practices**: Containerization, logging, health checks
- **Resource Management**: Shared volumes and network communication

Perfect for learning **modern AI/ML deployment** without complex infrastructure setup.

## Key Enhancements

### Recent Improvements

1. **Dependency Version Control**
   - Added `numpy<2.0` constraint to ensure compatibility with PyTorch and Transformers
   - Added `pyarrow<15.0.0` to fix dataset handling issues with newer versions

2. **Sequential Container Startup**
   - Implemented health checks for proper service orchestration
   - Added container startup dependencies for proper initialization order:
     ```
     scraper → preprocessor → trainer → summarizer → web-ui
     ```

3. **Dockerfile Enhancements**
   - Added curl installation for health check capabilities
   - Created startup scripts for proper service initialization

4. **Improved Error Handling & Timeout Management**
   - Extended API timeout to 120 seconds for model loading and inference
   - Better error messaging and user feedback
