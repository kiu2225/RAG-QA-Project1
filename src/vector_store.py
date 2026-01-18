"""
Vector store module for embedding generation and similarity search.
"""
import os
import pickle
from typing import List, Tuple
import numpy as np


class VectorStore:
    """Handles embedding generation and vector similarity search using FAISS."""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vector store.
        
        Args:
            embedding_model_name: Name of the sentence transformer model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(embedding_model_name)
        except ImportError:
            raise ImportError("sentence-transformers library is required. Install with: pip install sentence-transformers")
        
        self.index = None
        self.chunks = []
        self.dimension = None
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, chunks: List[str]):
        """
        Build a FAISS index from text chunks.
        
        Args:
            chunks: List of text chunks to index
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu library is required. Install with: pip install faiss-cpu")
        
        if not chunks:
            raise ValueError("Cannot build index from empty chunks list")
        
        print(f"Building index for {len(chunks)} chunks...")
        
        # Store chunks
        self.chunks = chunks
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        self.dimension = embeddings.shape[1]
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built successfully with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for most similar chunks to a query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples (chunk_text, distance_score)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distances[0][i])))
        
        return results
    
    def save(self, save_dir: str):
        """
        Save the vector store to disk.
        
        Args:
            save_dir: Directory to save the vector store
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu library is required")
        
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_dir, 'index.faiss'))
        
        # Save chunks
        with open(os.path.join(save_dir, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Vector store saved to {save_dir}")
    
    def load(self, save_dir: str):
        """
        Load the vector store from disk.
        
        Args:
            save_dir: Directory containing the saved vector store
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu library is required")
        
        # Load FAISS index
        index_path = os.path.join(save_dir, 'index.faiss')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load chunks
        chunks_path = os.path.join(save_dir, 'chunks.pkl')
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Vector store loaded from {save_dir}")
        print(f"Index contains {self.index.ntotal} vectors")
