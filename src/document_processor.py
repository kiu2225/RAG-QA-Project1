"""
Document processing module for loading and chunking text documents.
"""
import os
from typing import List


class DocumentProcessor:
    """Handles document loading and text chunking."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, directory_path: str) -> List[str]:
        """
        Load all text documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of document contents
        """
        documents = []
        
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Only process .txt files for simplicity
            if filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
        
        return documents
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Get chunk from start to start + chunk_size
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start forward, accounting for overlap
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def process_documents(self, directory_path: str) -> List[str]:
        """
        Load and chunk all documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of text chunks from all documents
        """
        documents = self.load_documents(directory_path)
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
