# RAG-QA-Project1

A Retrieval Augmented Generation (RAG) based Question Answering system built as a bootcamp project. This system demonstrates the core concepts of RAG by combining document retrieval with text generation to answer questions based on a knowledge base.

## Features

- **Document Processing**: Load and chunk text documents for efficient retrieval
- **Vector Embeddings**: Convert text to semantic embeddings using TF-IDF (or optional sentence transformers)
- **Vector Search**: Fast similarity search using cosine similarity
- **RAG Pipeline**: Retrieve relevant context and generate answers
- **Interactive Mode**: Chat-like interface for asking questions
- **Single Query Mode**: Process individual queries from command line
- **OpenAI Integration**: Optional integration with OpenAI GPT models for advanced answer generation

## Architecture

The system consists of four main components:

1. **Document Processor** (`src/document_processor.py`): Loads and chunks documents
2. **Vector Store** (`src/vector_store.py`): Creates embeddings and manages similarity search
3. **RAG Pipeline** (`src/rag_pipeline.py`): Orchestrates retrieval and generation
4. **Main Application** (`main.py`): CLI interface for the system

By default, the system uses TF-IDF vectorization which doesn't require downloading models. For better semantic understanding, you can use sentence transformers (requires internet to download models).

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kiu2225/RAG-QA-Project1.git
cd RAG-QA-Project1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up OpenAI API key for advanced answer generation:
```bash
# Create a .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

### Quick Start

1. Add your documents to the `documents/` directory (text files with `.txt` extension)

2. Run the system in interactive mode:
```bash
python main.py
```

The first run will process your documents and build the vector index. Subsequent runs will load the existing index.

### Command Line Options

```bash
# Interactive mode (default)
python main.py

# Single query mode
python main.py --query "What is machine learning?"

# Rebuild the index
python main.py --rebuild-index

# Custom documents directory
python main.py --documents-dir /path/to/documents

# Custom vector store directory
python main.py --vector-store-dir /path/to/vector_store
```

### Example Session

```
$ python main.py

Processing documents from documents...
Processed 15 chunks from documents
Building index for 15 chunks...
Index built successfully with 15 vectors
Vector store saved to vector_store

============================================================
RAG Question Answering System - Interactive Mode
============================================================
Type your questions below. Type 'quit' or 'exit' to stop.

Question: What is Python?

Processing query: What is Python?
Retrieved 3 relevant contexts

------------------------------------------------------------
Answer:
------------------------------------------------------------
Based on the retrieved information:

[Context 1]
Python is a high-level, interpreted programming language known 
for its simplicity and readability. Created by Guido van Rossum 
and first released in 1991, Python has become one of the most 
popular programming languages in the world.

...

Question: quit
Goodbye!
```

## Project Structure

```
RAG-QA-Project1/
├── documents/              # Document storage
│   ├── ai_ml_basics.txt
│   ├── python_basics.txt
│   └── vector_databases.txt
├── src/                    # Source code
│   ├── document_processor.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## How It Works

1. **Document Processing**: Text documents are loaded from the `documents/` directory and split into overlapping chunks
2. **Embedding Generation**: Each chunk is converted to a vector embedding using TF-IDF vectorization (or optionally sentence transformers)
3. **Index Building**: Embeddings are stored in memory as numpy arrays for cosine similarity search
4. **Query Processing**: When you ask a question:
   - The question is converted to an embedding using the same method
   - The most similar document chunks are retrieved using cosine similarity
   - The chunks are used as context to generate an answer
5. **Answer Generation**: 
   - With OpenAI: GPT model generates a natural language answer
   - Without OpenAI: The system returns the relevant context chunks

## Configuration

### Chunk Size and Overlap

Modify in `main.py`:
```python
doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
```

### Embedding Model

By default, the system uses TF-IDF:
```python
vector_store = VectorStore(embedding_model_name='tfidf')
```

For better semantic understanding with neural embeddings (requires internet):
```python
vector_store = VectorStore(embedding_model_name='all-MiniLM-L6-v2')
```

Other good models:
- `all-mpnet-base-v2` (better quality, slower)
- `all-MiniLM-L12-v2` (medium quality and speed)

Note: Neural embedding models require `sentence-transformers` and internet access to download.

### Number of Retrieved Contexts

Modify the `top_k` parameter when calling:
```python
result = rag_pipeline.query(query, top_k=5)  # Retrieve 5 contexts
```

## Adding Your Own Documents

1. Create text files (`.txt` extension) in the `documents/` directory
2. Run with `--rebuild-index` to reprocess:
```bash
python main.py --rebuild-index
```

## Dependencies

- `scikit-learn`: For TF-IDF vectorization and cosine similarity
- `numpy`: For numerical operations
- `openai`: (Optional) For GPT-based answer generation
- `python-dotenv`: For environment variable management
- `PyPDF2`: For future PDF support
- `sentence-transformers`: (Optional) For neural embeddings - requires internet to download models
- `faiss-cpu`: (Optional) For FAISS-based vector search with neural embeddings

## Future Enhancements

- [ ] PDF document support
- [ ] Support for multiple file formats (DOCX, MD, etc.)
- [ ] Web interface
- [ ] Document metadata tracking
- [ ] Advanced chunking strategies
- [ ] Hybrid search (keyword + semantic)
- [ ] Answer quality metrics
- [ ] Conversation history

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.