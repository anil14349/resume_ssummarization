"""RAG enhancement for resume summarization."""
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from pathlib import Path

class RAGEnhancer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize RAG enhancer with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.kb_path = Path("data/knowledge_base")
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.embeddings_file = self.kb_path / "embeddings.npz"
        self.documents_file = self.kb_path / "documents.json"
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load or initialize knowledge base."""
        if self.embeddings_file.exists() and self.documents_file.exists():
            # Load existing knowledge base
            self.embeddings = np.load(self.embeddings_file.as_posix())["embeddings"]
            with open(self.documents_file, "r") as f:
                self.documents = json.load(f)
        else:
            # Initialize empty knowledge base
            self.embeddings = np.array([])
            self.documents = []

    def save_knowledge_base(self):
        """Save knowledge base to disk."""
        np.savez(self.embeddings_file, embeddings=self.embeddings)
        with open(self.documents_file, "w") as f:
            json.dump(self.documents, f, indent=2)

    def add_to_knowledge_base(self, documents: List[Dict]):
        """Add new documents to knowledge base."""
        texts = [self._prepare_text(doc) for doc in documents]
        new_embeddings = self.model.encode(texts)
        
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.documents.extend(documents)
        self.save_knowledge_base()

    def enhance_prompt(self, resume_data: Dict, top_k: int = 3) -> str:
        """Enhance generation prompt with relevant examples."""
        query = self._prepare_text(resume_data)
        query_embedding = self.model.encode([query])[0]
        
        if len(self.embeddings) == 0:
            return ""
        
        # Find most similar documents
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Format examples
        examples = []
        for idx in top_indices:
            if similarities[idx] > 0.5:  # Only use if similarity is high enough
                doc = self.documents[idx]
                example = f"Example {len(examples)+1}:\n"
                example += f"Role: {doc.get('role', '')}\n"
                example += f"Summary: {doc.get('summary', '')}\n"
                examples.append(example)
        
        if not examples:
            return ""
        
        # Create context string
        context = "Here are some examples of professional summaries:\n\n"
        context += "\n".join(examples)
        context += "\n\nUsing these examples as inspiration, create a similar professional summary for:"
        
        return context

    def _prepare_text(self, data: Dict) -> str:
        """Prepare text for embedding."""
        parts = []
        if "role" in data:
            parts.append(f"Role: {data['role']}")
        if "current_role" in data:
            parts.append(f"Current Role: {data['current_role']}")
        if "skills" in data:
            parts.append(f"Skills: {data['skills']}")
        if "achievements" in data:
            parts.append(f"Achievements: {data['achievements']}")
        return " ".join(parts)
