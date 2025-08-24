import json, os
import cohere, numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Global state
model, documents, doc_embeddings, co = None, None, None, None

def Initializing():#init
    global model, documents, doc_embeddings, co
    if model: return
    
    print(" Initializing components...")
    load_dotenv()
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    print(" Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load or create embeddings
    if os.path.exists("embeddings.npy"):
        print(" Loading saved embeddings...")
        doc_embeddings = np.load("embeddings.npy")
        with open("chunks.json") as f: documents = json.load(f)
    else:
        print(" Loading documents...")
        with open("chunks.json") as f: documents = json.load(f)
        print(" Generating embeddings...")
        doc_embeddings = model.encode(documents)
        np.save("embeddings.npy", doc_embeddings)
    
    print(f" All components ready! ({len(documents)} documents)")

class RAGRequest(BaseModel):
    query: str
    top_n: int = 3

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/rag")
def rag_search(request: RAGRequest):
    Initializing()
    
    print(f" Processing query: {request.query}")
    
    # Step 1: Retrieval - Find similar documents
    print(" Generating query embedding...")
    query_emb = model.encode([request.query])#384-dimensional vector
    print(" Computing similarities...")
    similarities = cosine_similarity(query_emb, doc_embeddings)[0]#embedding similarit score
    top_indices = np.argsort(similarities)[::-1][:5] #Sort indices by similarity (ascending), Reverse to get descending order,Take top 5
    retrieved_docs = [documents[i] for i in top_indices]
    print(f" Retrieved {len(retrieved_docs)} documents")
    print(" Documents retrieved:", retrieved_docs)
    # Step 2: Augmentation - Rerank documents
    print(" Reranking with Cohere...")
    rerank_resp = co.rerank(
        query=request.query,
        documents=retrieved_docs,
        top_n=min(request.top_n, len(retrieved_docs)),# Return 3 documents,Safety check
        model="rerank-english-v3.0"
    )
    print(" Reranking complete")
    
    # Step 3: Generation - Create answer from context
    print(" Generating answer...")
    context = "\n".join([retrieved_docs[r.index] for r in rerank_resp.results])
    
    gen_resp = co.generate(
        prompt=f"Context: {context}\n\nQuestion: {request.query}\n\nAnswer:",
        model="command",
        max_tokens=200
    )
    print("Answer generated")
    print(" Returning complete RAG response")
    
    return {
        "query": request.query,
        "generated_answer": gen_resp.generations[0].text.strip(),
        "source_chunks": [
            {"rank": i+1, "text": retrieved_docs[r.index], "score": r.relevance_score}
            for i, r in enumerate(rerank_resp.results)
        ]
    }

# Mount static files after API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8002)