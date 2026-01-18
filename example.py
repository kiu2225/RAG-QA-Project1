#!/usr/bin/env python3
"""
Example script demonstrating how to use the RAG QA system programmatically.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


def main():
    """Example usage of the RAG system."""
    
    print("=" * 70)
    print("RAG QA System - Example Usage")
    print("=" * 70)
    
    # Step 1: Initialize components
    print("\n1. Initializing components...")
    doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    vector_store = VectorStore(embedding_model_name='tfidf')
    
    # Step 2: Process documents
    print("\n2. Processing documents...")
    documents_dir = 'documents'
    chunks = doc_processor.process_documents(documents_dir)
    print(f"   Processed {len(chunks)} chunks from documents")
    
    # Step 3: Build vector index
    print("\n3. Building vector index...")
    vector_store.build_index(chunks)
    print(f"   Index built with {len(chunks)} vectors")
    
    # Step 4: Initialize RAG pipeline
    print("\n4. Initializing RAG pipeline...")
    rag = RAGPipeline(vector_store, use_openai=False)
    
    # Step 5: Query the system
    print("\n5. Querying the system...")
    queries = [
        "What is artificial intelligence?",
        "What programming language should I learn?",
        "How do vector databases work?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: {query}")
        result = rag.query(query, top_k=2)
        
        print(f"\n   Retrieved {len(result['contexts'])} relevant contexts:")
        for j, ctx in enumerate(result['contexts'], 1):
            # Show first 100 characters of each context
            preview = ctx[:100].replace('\n', ' ') + "..."
            print(f"      Context {j}: {preview}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
