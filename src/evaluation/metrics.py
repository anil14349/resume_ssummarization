"""Evaluation metrics for generated summaries."""
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from textblob import TextBlob

class SummaryEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.tfidf = TfidfVectorizer()
        
    def calculate_rouge_scores(self, generated_summary, reference_summary):
        """Calculate ROUGE scores between generated and reference summaries."""
        if not reference_summary:
            return {
                'rouge1_f': 0.0,
                'rouge2_f': 0.0,
                'rougeL_f': 0.0
            }
            
        scores = self.rouge_scorer.score(reference_summary, generated_summary)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }
    
    def format_reference_text(self, input_data):
        """Format input data into a reference text."""
        lines = []
        
        # Add basic information
        if input_data.get('name'):
            lines.append(f"{input_data['name']}")
        if input_data.get('current_role'):
            lines.append(f"{input_data['current_role']}")
        if input_data.get('years_experience'):
            lines.append(f"{input_data['years_experience']} years of experience")
            
        # Add achievements
        if input_data.get('achievements'):
            lines.extend(input_data['achievements'])
                
        # Add skills
        if input_data.get('skills'):
            lines.extend(input_data['skills'])
                
        # Add education
        if input_data.get('education'):
            lines.extend(input_data['education'])
        
        return ' '.join(lines)
    
    def calculate_content_similarity(self, generated_summary, reference_text):
        """Calculate cosine similarity between generated summary and reference text."""
        try:
            # Create TF-IDF matrix
            tfidf_matrix = self.tfidf.fit_transform([generated_summary, reference_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            return 0.0
            
    def calculate_readability_metrics(self, text):
        """Calculate readability metrics for the text."""
        try:
            # Remove extra whitespace and normalize
            text = ' '.join(text.split())
            
            # Get sentence count
            sentences = len(TextBlob(text).sentences)
            if sentences == 0:
                return {
                    'avg_sentence_length': 0.0,
                    'sentiment_score': 0.0,
                    'subjectivity_score': 0.0
                }
            
            # Calculate average sentence length
            words = len(text.split())
            avg_sentence_length = words / sentences
            
            # Get sentiment and subjectivity scores
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            subjectivity_score = blob.sentiment.subjectivity
            
            return {
                'avg_sentence_length': avg_sentence_length,
                'sentiment_score': sentiment_score,
                'subjectivity_score': subjectivity_score
            }
        except Exception as e:
            print(f"Error in readability calculation: {e}")
            return {
                'avg_sentence_length': 0.0,
                'sentiment_score': 0.0,
                'subjectivity_score': 0.0
            }
    
    def evaluate_summary(self, generated_summary, reference_summary, input_data):
        """Evaluate the generated summary using local metrics."""
        # Calculate ROUGE scores if reference summary is available
        rouge_scores = self.calculate_rouge_scores(generated_summary, reference_summary)
        
        # Create reference text from input data
        reference_text = self.format_reference_text(input_data)
        
        # Calculate content similarity
        content_similarity = self.calculate_content_similarity(generated_summary, reference_text)
        
        # Calculate readability metrics
        readability_scores = self.calculate_readability_metrics(generated_summary)
        
        return {
            **rouge_scores,
            'content_similarity': content_similarity,
            **readability_scores
        }
