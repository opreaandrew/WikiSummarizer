import os
import json
import time
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import logging

# Configure basic service logging
logging.basicConfig(
    level=logging.INFO,
    format="[trainer] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("trainer")

class SummarizationTrainer:
    def __init__(self, data_dir="/app/data", model_dir="/app/models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Use a lightweight model for training
        self.model_name = "facebook/bart-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def load_preprocessed_data(self, preprocessed_file):
        """Load preprocessed training data"""
        training_data = []
        
        with open(preprocessed_file, 'r', encoding='utf-8') as f:
            for line in f:
                training_data.append(json.loads(line.strip()))
        
        return training_data
    
    def tokenize_data(self, examples):
        """Tokenize the training data"""
        model_inputs = self.tokenizer(
            examples["input_text"], 
            max_length=512, 
            truncation=True, 
            padding=True
        )
        
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["target_text"], 
                max_length=128, 
                truncation=True, 
                padding=True
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def train_model(self, preprocessed_file, topic):
        """Train the summarization model"""
        print(f"Starting training for topic: {topic}")
        
        try:
            # Load data
            training_data = self.load_preprocessed_data(preprocessed_file)
            print(f"Loaded {len(training_data)} training examples")
            
            # Create dataset
            dataset = Dataset.from_list(training_data)
            tokenized_dataset = dataset.map(
                self.tokenize_data, 
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Load model
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Training arguments - lightweight for demo purposes
            training_args = TrainingArguments(
                output_dir=str(self.model_dir / f"{topic}_model"),
                num_train_epochs=1,  # Very light training
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                warmup_steps=10,
                learning_rate=5e-5,
                logging_steps=5,
                save_steps=50,
                eval_steps=50,
                save_total_limit=1,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to=None,
            )
            
            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=model,
                padding=True
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
            
            # Train the model
            print("Starting model training...")
            trainer.train()
            
            # Save the model
            model_save_path = self.model_dir / f"{topic}_model"
            trainer.save_model(str(model_save_path))
            self.tokenizer.save_pretrained(str(model_save_path))
            
            print(f"Model training completed and saved to: {model_save_path}")
            return model_save_path
            
        except Exception as e:
            print(f"Error during training: {e}")
            return None

def main():
    """Main function to handle training requests"""
    trainer = SummarizationTrainer()
    
    # Check for training requests
    preprocess_complete_file = Path("/app/data/preprocess_complete.txt")
    
    while True:
        if preprocess_complete_file.exists():
            try:
                with open(preprocess_complete_file, 'r') as f:
                    preprocessed_file_path = f.read().strip()
                
                if preprocessed_file_path and Path(preprocessed_file_path).exists():
                    # Extract topic from filename
                    topic = Path(preprocessed_file_path).stem.replace('_preprocessed', '')
                    
                    print(f"Starting training for: {topic}")
                    result = trainer.train_model(preprocessed_file_path, topic)
                    
                    # Create completion marker for summarizer
                    if result:
                        train_complete_file = Path("/app/data/train_complete.txt")
                        with open(train_complete_file, 'w') as f:
                            f.write(f"{topic}|{result}")
                    
                    # Remove preprocessing completion file
                    preprocess_complete_file.unlink()
                    
            except Exception as e:
                print(f"Error processing training request: {e}")
                preprocess_complete_file.unlink(missing_ok=True)
        
        time.sleep(3)  # Check every 3 seconds

if __name__ == "__main__":
    main()
