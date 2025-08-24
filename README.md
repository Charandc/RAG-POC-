# AI RAG Search System

A modern **Retrieval-Augmented Generation (RAG)** system built with FastAPI and Cohere AI that provides intelligent document search and answer generation.

## Features

- **Semantic Search** - Find relevant documents using AI embeddings
- **AI Answer Generation** - Get intelligent responses using Cohere's language models
- **Document Reranking** - Advanced relevance scoring for better results
- **Futuristic UI** - Modern dark theme with animations and gradients
- **Fast Performance** - Cached embeddings and optimized retrieval
- **Responsive Design** - Works on desktop and mobile devices

## Tech Stack

- **Backend**: FastAPI, Python
- **AI Models**: Cohere API, SentenceTransformers
- **Frontend**: HTML, CSS, JavaScript
- **Data**: NumPy, JSON

## Prerequisites

- Python 3.8+
- Cohere API Key (free at [cohere.ai](https://cohere.ai))

## Quick Start

### 1. Clone & Navigate
```bash
cd RAG
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn cohere numpy python-dotenv sentence-transformers scikit-learn
```

### 3. Set Up Environment
Create `.env` file:
```env
COHERE_API_KEY=your_cohere_api_key_here
```

### 4. Run the Application
```bash
python main.py
```

### 5. Open Your Browser
Visit: `http://localhost:8002`

## Project Structure

```
RAG/
├── main.py              # FastAPI backend server
├── static/
│   └── index.html       # Futuristic web interface
├── chunks.json          # Document knowledge base (56 European city facts)
├── embeddings.npy       # Cached document embeddings (auto-generated)
├── .env                 # API keys (create this)
└── README.md           # This file
```

## API Endpoints

### Search Documents
```http
POST /rag
Content-Type: application/json

{
  "query": "What is the capital of France?",
  "top_n": 3
}
```

### Health Check
```http
GET /health
```

## Usage Examples

Try these queries:
- "What is the capital of France?"
- "Tell me about Paris landmarks"
- "Which cities are on rivers?"
- "What museums are in Amsterdam?"

## How It Works

1. **Retrieval**: Converts your query to embeddings and finds similar documents
2. **Augmentation**: Reranks documents using Cohere's reranking model
3. **Generation**: Creates intelligent answers using retrieved context

## System Flow

```
User Query → Embedding → Similarity Search → Reranking → AI Generation → Response
```

## Configuration

### Modify Search Parameters
Edit in `main.py`:
```python
top_indices = np.argsort(similarities)[::-1][:5]  # Change 5 to get more docs
top_n=min(request.top_n, len(retrieved_docs))     # Adjust reranking count
max_tokens=200                                     # Change answer length
```

### Add Your Own Documents
Replace content in `chunks.json` with your own text chunks, then delete `embeddings.npy` to regenerate embeddings.

## Troubleshooting

### Common Issues:

**Port already in use:**
```bash
# Change port in main.py
uvicorn.run(app, port=8003)  # Use different port
```

**Missing API key:**
- Ensure `.env` file exists with valid Cohere API key
- Get free key at [cohere.ai](https://cohere.ai)

**Dependencies error:**
```bash
pip install --upgrade pip
pip install -r requirements.txt  # If you create one
```

## Performance

- **First run**: ~10-15 seconds (downloads model + generates embeddings)
- **Subsequent runs**: ~2-3 seconds per query
- **Memory usage**: ~500MB (model + embeddings)

## Future Enhancements

- [ ] Document upload functionality
- [ ] Multiple file format support (PDF, DOCX)
- [ ] User authentication
- [ ] Chat history
- [ ] Advanced filtering options

## License

MIT License - Feel free to use and modify!

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

