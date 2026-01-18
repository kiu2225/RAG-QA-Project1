"""
RAG (Retrieval Augmented Generation) pipeline module.
"""
import os
from typing import List, Optional
from dotenv import load_dotenv


class RAGPipeline:
    """RAG pipeline that combines retrieval and generation."""
    
    def __init__(self, vector_store, use_openai: bool = False):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: VectorStore instance for retrieval
            use_openai: Whether to use OpenAI API for generation (requires API key)
        """
        self.vector_store = vector_store
        self.use_openai = use_openai
        self.openai_client = None
        
        if use_openai:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai library is required. Install with: pip install openai")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant text chunks
        """
        results = self.vector_store.search(query, top_k=top_k)
        contexts = [chunk for chunk, _ in results]
        return contexts
    
    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate an answer using retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        if self.use_openai and self.openai_client:
            return self._generate_with_openai(query, contexts)
        else:
            return self._generate_simple(query, contexts)
    
    def _generate_with_openai(self, query: str, contexts: List[str]) -> str:
        """Generate answer using OpenAI API."""
        # Build context string
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        # Create prompt
        prompt = f"""Answer the question based on the context below. If the question cannot be answered using the information provided, say "I don't have enough information to answer this question."

Context:
{context_str}

Question: {query}

Answer:"""
        
        # Call OpenAI API
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _generate_simple(self, query: str, contexts: List[str]) -> str:
        """Generate a simple answer by returning relevant contexts."""
        if not contexts:
            return "No relevant information found to answer the query."
        
        answer = f"Based on the retrieved information:\n\n"
        for i, ctx in enumerate(contexts, 1):
            answer += f"[Context {i}]\n{ctx}\n\n"
        
        answer += f"\nNote: This is a simple retrieval-based response. "
        answer += f"For AI-generated answers, set OPENAI_API_KEY environment variable and use use_openai=True."
        
        return answer
    
    def query(self, query: str, top_k: int = 3) -> dict:
        """
        Process a query end-to-end.
        
        Args:
            query: User query
            top_k: Number of context chunks to retrieve
            
        Returns:
            Dictionary containing query, contexts, and answer
        """
        print(f"\nProcessing query: {query}")
        
        # Retrieve contexts
        contexts = self.retrieve_context(query, top_k=top_k)
        print(f"Retrieved {len(contexts)} relevant contexts")
        
        # Generate answer
        answer = self.generate_answer(query, contexts)
        
        return {
            'query': query,
            'contexts': contexts,
            'answer': answer
        }
