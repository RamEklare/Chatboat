from llama_cpp import Llama

class LocalLLaMAModel:
    def __init__(self, model_path=r"C:\Users\HP\Bajaj_techno\model\Llama-3-8B-Instruct-Finance-RAG.Q4_K_S.gguf", context_size=2048):
        self.llm = Llama(model_path=model_path, n_ctx=context_size)

    def generate(self, prompt, max_tokens=256, stop=["\nUser:", "\n"]):
        response = self.llm(prompt, max_tokens=max_tokens, stop=stop)
        return response['choices'][0]['text'].strip()
