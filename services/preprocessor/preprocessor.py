import os
import warnings
import logging

# Configure basic service logging
logging.basicConfig(
    level=logging.INFO,
    format="[preprocessor] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("preprocessor")

# Disable the specific warning about missing PyTorch
warnings.filterwarnings("ignore", message="None of PyTorch")

import json
import time
import re
import nltk
from pathlib import Path
from transformers import AutoTokenizer

class TextPreprocessor:
    def __init__(self, data_dir="/app/data"):
        self.data_dir = Path(data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove citations like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short sentences (less than 10 characters)
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return ' '.join(sentences)
    
    def create_training_pairs(self, articles, max_input_length=512, max_target_length=128):
        """Create input-output pairs for summarization training"""
        training_data = []
        
        for article in articles:
            content = self.clean_text(article['content'])
            summary = self.clean_text(article['summary'])
            
            # Split content into chunks that fit the model's input size
            sentences = nltk.sent_tokenize(content)
            
            current_chunk = ""
            for sentence in sentences:
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                # Check if tokenized length is within limits
                tokens = self.tokenizer.encode(test_chunk, truncation=False)
                
                if len(tokens) <= max_input_length:
                    current_chunk = test_chunk
                else:
                    # Process current chunk if it's substantial
                    if len(current_chunk.split()) > 20:  # At least 20 words
                        training_data.append({
                            "input_text": current_chunk.strip(),
                            "target_text": summary[:max_target_length * 4],  # Rough character limit
                            "title": article['title']
                        })
                    current_chunk = sentence
            
            # Don't forget the last chunk
            if len(current_chunk.split()) > 20:
                training_data.append({
                    "input_text": current_chunk.strip(),
                    "target_text": summary[:max_target_length * 4],
                    "title": article['title']
                })
        
        return training_data
    
    def preprocess_scraped_data(self, scraped_file):
        """Process scraped Wikipedia data into training format"""
        logger.info(f"Preprocessing scraped data from: {scraped_file}")
        
        try:
            with open(scraped_file, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
            
            # Create training pairs
            training_data = self.create_training_pairs(scraped_data['articles'])
            
            # Save preprocessed data
            topic = scraped_data['topic'].replace(' ', '_')
            output_file = self.data_dir / f"{topic}_preprocessed.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Preprocessed {len(training_data)} training examples")
            logger.info(f"Preprocessed data saved to: {output_file}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None

def main():
    """Main function to handle preprocessing requests"""
    preprocessor = TextPreprocessor()
    
    # Check for preprocessing requests
    completion_file = Path("/app/data/scrape_complete.txt")
    
    while True:
        if completion_file.exists():
            try:
                with open(completion_file, 'r') as f:
                    scraped_file_path = f.read().strip()
                
                if scraped_file_path and Path(scraped_file_path).exists():
                    logger.info(f"Processing scraped data: {scraped_file_path}")
                    result = preprocessor.preprocess_scraped_data(scraped_file_path)
                    
                    # Create completion marker for trainer
                    if result:
                        preprocess_complete_file = Path("/app/data/preprocess_complete.txt")
                        with open(preprocess_complete_file, 'w') as f:
                            f.write(str(result))
                    
                    # Remove scrape completion file
                    completion_file.unlink()
                    
            except Exception as e:
                logger.error(f"Error processing scraped data: {e}")
                completion_file.unlink(missing_ok=True)
        
        time.sleep(2)  # Check every 2 seconds

if __name__ == "__main__":
    main()
