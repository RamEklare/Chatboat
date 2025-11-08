class RAGChatbot:
    def __init__(self, vector_indexer, llm_model):
        self.vector_indexer = vector_indexer
        self.llm_model = llm_model

    def chat(self, query):
        relevant_docs = self.vector_indexer.query(query, top_k=4)
        context = "\n\n".join(relevant_docs)
        prompt = (
            f"Answer the question using ONLY the information below. Cite facts from the context whenever possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        answer = self.llm_model.generate(prompt)
        return answer
