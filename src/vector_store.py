"""
Vector store module for embedding generation and similarity search.
"""
import os
import pickle
from typing import List, Tuple
import numpy as np


class VectorStore:
    """Handles embedding generation and vector similarity search using TF-IDF."""
    
    def __init__(self, embedding_model_name: str = 'tfidf'):
        """
        Initialize the vector store.
        
        Args:
            embedding_model_name: Type of embeddings to use ('tfidf' or 'sentence-transformers')
        """
        self.embedding_model_name = embedding_model_name
        self.encoder = None
        self.use_transformers = False
        
        if embedding_model_name == 'tfidf':
            # Use TF-IDF vectorizer (no internet required)
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.encoder = TfidfVectorizer(max_features=1000, stop_words='english')
        else:
            # Try to use sentence transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(embedding_model_name)
                self.use_transformers = True
            except ImportError:
                raise ImportError("sentence-transformers library is required for neural embeddings. "
                                "Install with: pip install sentence-transformers")
        
        self.chunks = []
        self.embeddings = None
        self.dimension = None
    
    def create_embeddings(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            fit: Whether to fit the vectorizer (for TF-IDF)
            
        Returns:
            Numpy array of embeddings
        """
        if self.use_transformers:
            # Use sentence transformers
            embeddings = self.encoder.encode(texts, show_progress_bar=True)
            return embeddings
        else:
            # Use TF-IDF
            if fit:
                embeddings = self.encoder.fit_transform(texts).toarray()
            else:
                embeddings = self.encoder.transform(texts).toarray()
            return embeddings
    
    def build_index(self, chunks: List[str]):
        """
        Build an index from text chunks.
        
        Args:
            chunks: List of text chunks to index
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunks list")
        
        print(f"Building index for {len(chunks)} chunks...")
        
        # Store chunks
        self.chunks = chunks
        
        # Create embeddings
        self.embeddings = self.create_embeddings(chunks, fit=True)
        self.dimension = self.embeddings.shape[1]
        
        print(f"Index built successfully with {len(self.chunks)} vectors (dimension: {self.dimension})")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for most similar chunks to a query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples (chunk_text, similarity_score)
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.create_embeddings([query], fit=False)
        
        # Compute cosine similarity
        # Normalize vectors
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        docs_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute similarities
        similarities = np.dot(docs_norm, query_norm.T).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(similarities[idx])))
        
        return results
    
    def save(self, save_dir: str):
        """
        Save the vector store to disk.
        
        Args:
            save_dir: Directory to save the vector store
        """
        if self.embeddings is None:
            raise ValueError("No index to save. Build index first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save embeddings and chunks
        with open(os.path.join(save_dir, 'vector_store.pkl'), 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'chunks': self.chunks,
                'encoder': self.encoder,
                'embedding_model_name': self.embedding_model_name
            }, f)
        
        print(f"Vector store saved to {save_dir}")
    
    def load(self, save_dir: str):
        """
        Load the vector store from disk.
        
        Args:
            save_dir: Directory containing the saved vector store
        """
        # Load vector store
        store_path = os.path.join(save_dir, 'vector_store.pkl')
        if not os.path.exists(store_path):
            raise FileNotFoundError(f"Vector store file not found: {store_path}")
        
        with open(store_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.chunks = data['chunks']
        self.encoder = data['encoder']
        self.embedding_model_name = data['embedding_model_name']
        self.dimension = self.embeddings.shape[1]
        
        print(f"Vector store loaded from {save_dir}")
        print(f"Index contains {len(self.chunks)} vectors (dimension: {self.dimension})")
