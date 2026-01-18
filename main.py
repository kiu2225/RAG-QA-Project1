"""
Main application for RAG-based Question Answering system.
"""
import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


def setup_rag_system(documents_dir: str, vector_store_dir: str = 'vector_store', 
                     rebuild_index: bool = False) -> RAGPipeline:
    """
    Set up the RAG system by processing documents and building/loading index.
    
    Args:
        documents_dir: Directory containing documents to process
        vector_store_dir: Directory to save/load vector store
        rebuild_index: Whether to rebuild the index even if it exists
        
    Returns:
        Initialized RAGPipeline instance
    """
    # Initialize components
    doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    # Use 'tfidf' for simple embeddings that don't require internet
    # Or use 'all-MiniLM-L6-v2' for neural embeddings if you have internet access
    vector_store = VectorStore(embedding_model_name='tfidf')
    
    # Check if we should load existing index
    if not rebuild_index and os.path.exists(vector_store_dir):
        print(f"Loading existing vector store from {vector_store_dir}...")
        try:
            vector_store.load(vector_store_dir)
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Rebuilding index...")
            rebuild_index = True
    else:
        rebuild_index = True
    
    # Build new index if needed
    if rebuild_index:
        print(f"Processing documents from {documents_dir}...")
        chunks = doc_processor.process_documents(documents_dir)
        
        if not chunks:
            raise ValueError(f"No documents found in {documents_dir}. Please add .txt files.")
        
        print(f"Processed {len(chunks)} chunks from documents")
        
        # Build vector store
        vector_store.build_index(chunks)
        
        # Save vector store
        vector_store.save(vector_store_dir)
    
    # Initialize RAG pipeline
    # Set use_openai=True if you have OPENAI_API_KEY set in .env file
    use_openai = os.getenv('OPENAI_API_KEY') is not None
    rag_pipeline = RAGPipeline(vector_store, use_openai=use_openai)
    
    return rag_pipeline


def interactive_mode(rag_pipeline: RAGPipeline):
    """
    Run the RAG system in interactive mode.
    
    Args:
        rag_pipeline: Initialized RAGPipeline instance
    """
    print("\n" + "="*60)
    print("RAG Question Answering System - Interactive Mode")
    print("="*60)
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            query = input("Question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Process query
            result = rag_pipeline.query(query, top_k=3)
            
            print("\n" + "-"*60)
            print("Answer:")
            print("-"*60)
            print(result['answer'])
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def single_query_mode(rag_pipeline: RAGPipeline, query: str):
    """
    Process a single query and exit.
    
    Args:
        rag_pipeline: Initialized RAGPipeline instance
        query: Query to process
    """
    result = rag_pipeline.query(query, top_k=3)
    
    print("\n" + "="*60)
    print("Query Results")
    print("="*60)
    print(f"\nQuestion: {result['query']}\n")
    print("-"*60)
    print("Answer:")
    print("-"*60)
    print(result['answer'])
    print("\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='RAG-based Question Answering System')
    parser.add_argument('--documents-dir', type=str, default='documents',
                        help='Directory containing documents to process (default: documents)')
    parser.add_argument('--vector-store-dir', type=str, default='vector_store',
                        help='Directory to save/load vector store (default: vector_store)')
    parser.add_argument('--rebuild-index', action='store_true',
                        help='Rebuild the index even if it exists')
    parser.add_argument('--query', type=str, default=None,
                        help='Single query to process (interactive mode if not provided)')
    
    args = parser.parse_args()
    
    try:
        # Setup RAG system
        rag_pipeline = setup_rag_system(
            documents_dir=args.documents_dir,
            vector_store_dir=args.vector_store_dir,
            rebuild_index=args.rebuild_index
        )
        
        # Run in appropriate mode
        if args.query:
            single_query_mode(rag_pipeline, args.query)
        else:
            interactive_mode(rag_pipeline)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
