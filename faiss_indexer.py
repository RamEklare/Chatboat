from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class FAISSIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        self.chunks = chunks
        embeddings = self.model.encode(chunks).astype('float32')
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def query(self, query_text, top_k=5):
        query_embedding = self.model.encode([query_text]).astype('float32')
        D, I = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in I[0]]
