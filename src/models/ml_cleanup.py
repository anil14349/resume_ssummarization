"""ML-based cleanup for model outputs."""
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

class SummaryCleanupDataset(Dataset):
    """Dataset for training cleanup models."""
    
    def __init__(self, examples: List[Tuple[str, str]], tokenizer):
        """Initialize dataset with raw->clean pairs."""
        self.examples = examples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        raw, clean = self.examples[idx]
        inputs = self.tokenizer(
            raw,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            clean,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }

class MLCleanupEnhancer:
    """ML-based cleanup using multiple models."""
    
    def __init__(
        self,
        quality_model: str = "microsoft/deberta-v3-small",
        cleanup_model: str = "t5-small",
        style_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        """Initialize cleanup models.
        
        Args:
            quality_model: Model for quality scoring
            cleanup_model: Model for seq2seq cleanup
            style_model: Model for style matching
            device: Device to run models on (cuda/cpu)
        """
        logger.info("Initializing ML Cleanup Enhancer")
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Quality scoring model
            logger.debug(f"Loading quality model: {quality_model}")
            self.quality_tokenizer = AutoTokenizer.from_pretrained(quality_model)
            self.quality_model = AutoModelForSequenceClassification.from_pretrained(
                quality_model,
                num_labels=2
            ).to(self.device)
            
            # Seq2seq cleanup model
            logger.debug(f"Loading cleanup model: {cleanup_model}")
            self.cleanup_tokenizer = T5Tokenizer.from_pretrained(cleanup_model)
            self.cleanup_model = T5ForConditionalGeneration.from_pretrained(
                cleanup_model
            ).to(self.device)
            
            # Style matching model
            logger.debug(f"Loading style model: {style_model}")
            self.style_model = SentenceTransformer(style_model).to(self.device)
            
            logger.info("Successfully loaded all models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
            
        # Save paths for training data and checkpoints
        self.data_dir = Path("data/cleanup")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def filter_sentences(
        self,
        sentences: List[str],
        threshold: float = 0.7
    ) -> List[str]:
        """Filter sentences based on quality score."""
        logger.debug(f"Filtering {len(sentences)} sentences")
        
        try:
            scores = []
            for sent in sentences:
                inputs = self.quality_tokenizer(
                    sent,
                    return_tensors="pt",
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.quality_model(**inputs)
                    score = outputs.logits.softmax(dim=-1)[0][1].item()
                    scores.append(score)
                    
            # Filter sentences above threshold
            filtered = [
                s for s, score in zip(sentences, scores)
                if score > threshold
            ]
            
            logger.debug(f"Kept {len(filtered)}/{len(sentences)} sentences")
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering sentences: {e}")
            raise
            
    def rank_by_style(
        self,
        sentences: List[str],
        style_example: str,
        top_k: int = 3
    ) -> List[str]:
        """Rank sentences by similarity to style example."""
        logger.debug(f"Ranking {len(sentences)} sentences")
        
        try:
            # Get embeddings
            target_embedding = self.style_model.encode(
                [style_example],
                convert_to_tensor=True
            )
            sent_embeddings = self.style_model.encode(
                sentences,
                convert_to_tensor=True
            )
            
            # Calculate similarities
            similarities = cosine_similarity(
                sent_embeddings.cpu(),
                target_embedding.cpu()
            )
            
            # Sort by similarity
            ranked_pairs = sorted(
                zip(sentences, similarities),
                key=lambda x: x[1].item(),
                reverse=True
            )
            
            # Take top k
            top_sentences = [s for s, _ in ranked_pairs[:top_k]]
            logger.debug(f"Selected top {len(top_sentences)} sentences")
            return top_sentences
            
        except Exception as e:
            logger.error(f"Error ranking sentences: {e}")
            raise
            
    def seq2seq_cleanup(self, text: str) -> str:
        """Clean text using seq2seq model."""
        logger.debug("Performing seq2seq cleanup")
        
        try:
            # Prepare input
            input_text = f"clean summary: {text}"
            inputs = self.cleanup_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.cleanup_model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
                
            # Decode
            cleaned = self.cleanup_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            logger.debug(f"Cleaned text: {cleaned}")
            return cleaned
            
        except Exception as e:
            logger.error(f"Error in seq2seq cleanup: {e}")
            raise
            
    def clean_output(
        self,
        text: str,
        style_example: Optional[str] = None
    ) -> str:
        """Clean model output using ML pipeline."""
        logger.info("Starting ML cleanup")
        logger.debug(f"Input text: {text}")
        
        try:
            # Split into sentences
            sentences = sent_tokenize(text)
            logger.debug(f"Split into {len(sentences)} sentences")
            
            # Filter low quality sentences
            quality_sentences = self.filter_sentences(sentences)
            
            # Rank by style if example provided
            if style_example and quality_sentences:
                ranked_sentences = self.rank_by_style(
                    quality_sentences,
                    style_example
                )
            else:
                ranked_sentences = quality_sentences[:3]
                
            # Join sentences
            combined = " ".join(ranked_sentences)
            
            # Final seq2seq cleanup
            final_summary = self.seq2seq_cleanup(combined)
            
            logger.info("Cleanup complete")
            logger.debug(f"Final summary: {final_summary}")
            return final_summary
            
        except Exception as e:
            logger.error(f"Error in cleanup pipeline: {e}")
            raise
            
    def save(self, path: str):
        """Save models and tokenizers."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save quality model
        self.quality_model.save_pretrained(save_path / "quality")
        self.quality_tokenizer.save_pretrained(save_path / "quality")
        
        # Save cleanup model
        self.cleanup_model.save_pretrained(save_path / "cleanup")
        self.cleanup_tokenizer.save_pretrained(save_path / "cleanup")
        
        # Save style model
        self.style_model.save(str(save_path / "style"))
        
    @classmethod
    def load(cls, path: str):
        """Load saved models."""
        path = Path(path)
        
        instance = cls(
            quality_model=str(path / "quality"),
            cleanup_model=str(path / "cleanup"),
            style_model=str(path / "style")
        )
        return instance
        
    def train(
        self,
        examples: List[Tuple[str, str]],
        eval_examples: Optional[List[Tuple[str, str]]] = None,
        **kwargs
    ):
        """Fine-tune the cleanup models on examples."""
        logger.info(f"Starting training with {len(examples)} examples")
        
        try:
            # Create datasets
            train_dataset = SummaryCleanupDataset(
                examples,
                self.cleanup_tokenizer
            )
            
            eval_dataset = None
            if eval_examples:
                eval_dataset = SummaryCleanupDataset(
                    eval_examples,
                    self.cleanup_tokenizer
                )
                
            # Training arguments
            training_args = TrainingArguments(
                output_dir="checkpoints/cleanup",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="logs",
                **kwargs
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.cleanup_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
            
            # Train model
            trainer.train()
            
            logger.info("Training complete")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
