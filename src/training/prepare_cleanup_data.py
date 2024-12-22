"""Prepare training data for ML cleanup from resume templates and dataset."""
import logging
from pathlib import Path
import pandas as pd
from docx import Document
import json
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

logger = logging.getLogger(__name__)

class CleanupDataPreparator:
    """Prepare training data for ML cleanup model."""
    
    def __init__(self, templates_dir: str, dataset_path: str):
        """Initialize data preparator.
        
        Args:
            templates_dir: Directory containing template docx files
            dataset_path: Path to CSV file with resume text data
        """
        self.templates_dir = Path(templates_dir)
        self.dataset_path = Path(dataset_path)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
    def read_docx_templates(self) -> List[str]:
        """Read and extract text from template docx files."""
        templates = []
        
        for docx_file in self.templates_dir.glob("*.docx"):
            try:
                doc = Document(docx_file)
                text = "\n".join([para.text for para in doc.paragraphs if para.text])
                templates.append(text)
                logger.debug(f"Read template: {docx_file}")
            except Exception as e:
                logger.error(f"Error reading {docx_file}: {e}")
                
        return templates
        
    def read_dataset(self) -> List[str]:
        """Read resume text from CSV dataset."""
        try:
            df = pd.read_csv(self.dataset_path)
            resumes = df['resume'].tolist()
            logger.debug(f"Read {len(resumes)} resumes from dataset")
            return resumes
        except Exception as e:
            logger.error(f"Error reading dataset: {e}")
            raise
            
    def evaluate_summary(self, generated: str, reference: str) -> Dict:
        """Calculate evaluation metrics for summary quality.
        
        Args:
            generated: Generated/cleaned summary
            reference: Reference/gold summary
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Tokenize
            gen_tokens = nltk.word_tokenize(generated.lower())
            ref_tokens = nltk.word_tokenize(reference.lower())
            
            # BLEU score
            bleu = sentence_bleu([ref_tokens], gen_tokens)
            
            # METEOR score
            meteor = meteor_score([ref_tokens], gen_tokens)
            
            # ROUGE scores
            rouge_scores = self.scorer.score(generated, reference)
            
            # BERTScore
            P, R, F1 = bert_score([generated], [reference], lang='en')
            bert_f1 = F1.item()
            
            metrics = {
                'bleu': bleu,
                'meteor': meteor,
                'rouge1_f': rouge_scores['rouge1'].fmeasure,
                'rouge2_f': rouge_scores['rouge2'].fmeasure,
                'rougeL_f': rouge_scores['rougeL'].fmeasure,
                'bert_score': bert_f1
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
            
    def prepare_training_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Prepare training and validation data.
        
        Returns:
            Tuple of (training_examples, validation_examples)
        """
        try:
            # Read data
            templates = self.read_docx_templates()
            resumes = self.read_dataset()
            
            # Combine data
            all_examples = []
            
            # Create pairs using templates as clean examples
            for i, template in enumerate(templates):
                for resume in resumes[:5]:  # Use first 5 resumes for each template
                    all_examples.append((resume, template))
                    
            # Split into train/val
            train_examples, val_examples = train_test_split(
                all_examples,
                test_size=0.2,
                random_state=42
            )
            
            logger.info(f"Prepared {len(train_examples)} training and {len(val_examples)} validation examples")
            
            # Save examples
            self.save_examples(train_examples, val_examples)
            
            return train_examples, val_examples
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
            
    def save_examples(self, train_examples: List[Tuple], val_examples: List[Tuple]):
        """Save prepared examples to disk."""
        data_dir = Path("data/cleanup")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training data
        with open(data_dir / "train_examples.json", "w") as f:
            json.dump({"examples": train_examples}, f)
            
        # Save validation data
        with open(data_dir / "val_examples.json", "w") as f:
            json.dump({"examples": val_examples}, f)
            
        logger.info("Saved examples to disk")
        
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize preparator
    preparator = CleanupDataPreparator(
        templates_dir="src/templates",
        dataset_path="updatedResumeDataset.csv"
    )
    
    # Prepare data
    train_examples, val_examples = preparator.prepare_training_data()
