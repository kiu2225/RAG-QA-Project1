# Quick Start Guide

## What is this project?

This is a **Retrieval Augmented Generation (RAG)** based Question Answering system - a bootcamp project that demonstrates the core concepts of RAG by combining document retrieval with text generation.

## What can it do?

- Answer questions based on documents in your knowledge base
- Retrieve relevant context from your documents
- Generate answers using the retrieved context
- Optionally use OpenAI GPT for advanced natural language answers

## How to get started in 3 steps:

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the system
```bash
python main.py
```

### Step 3: Ask questions!
```
Question: What is Python?
Question: What is machine learning?
Question: quit
```

## Example Queries

Try asking:
- "What is artificial intelligence?"
- "What are vector databases used for?"
- "What programming language should I learn?"
- "Explain deep learning"
- "What are popular Python libraries?"

## Adding Your Own Documents

1. Create text files (`.txt`) in the `documents/` directory
2. Run with rebuild flag: `python main.py --rebuild-index`
3. Start asking questions about your documents!

## How It Works

```
Your Question → TF-IDF Vectorization → Similarity Search → Retrieve Top 3 Chunks → Generate Answer
```

1. **Your documents** are split into chunks and converted to vectors
2. **Your question** is converted to a vector using the same method
3. **Most similar chunks** are found using cosine similarity
4. **Answer is generated** from the retrieved chunks

## Advanced Usage

### Single Query Mode
```bash
python main.py --query "What is Python?"
```

### Programmatic Usage
```python
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

# Initialize
vector_store = VectorStore(embedding_model_name='tfidf')
vector_store.load('vector_store')

# Create pipeline
rag = RAGPipeline(vector_store, use_openai=False)

# Query
result = rag.query("What is Python?", top_k=3)
print(result['answer'])
```

### Using OpenAI for Better Answers

1. Get an API key from OpenAI
2. Create a `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   ```
3. Run the system - it will automatically use GPT for answers!

## Project Structure

```
RAG-QA-Project1/
├── documents/          # Your knowledge base (text files)
├── src/               # Source code modules
├── main.py            # Main CLI application
├── example.py         # Example programmatic usage
└── README.md          # Full documentation
```

## Need Help?

See the full [README.md](README.md) for:
- Detailed architecture explanation
- Configuration options
- Troubleshooting
- Advanced features

## Technologies Used

- **scikit-learn**: TF-IDF vectorization
- **NumPy**: Vector operations and cosine similarity
- **OpenAI** (optional): Advanced answer generation
- **Python 3.7+**: Core implementation

---

**Ready to learn more?** Check out the complete README.md file for full documentation!
