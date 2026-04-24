import requests

class LLMGenerator:
    def __init__(self, model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, query, context):
        prompt = f"""
You are an expert AI assistant.

Answer the question using ONLY the context below.
Do NOT copy text directly. Explain clearly.

If the answer is not in the context, say: "Not found in context."

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = requests.post(
                self.url,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=60,
            )

            if response.status_code != 200:
                print("[HTTP ERROR]", response.status_code, response.text)
                return "LLM Error: HTTP request failed."

            data = response.json()

            if "response" not in data:
                print("[OLLAMA ERROR RESPONSE]", data)
                return "LLM Error: Invalid response format."

            return data["response"].strip()

        except requests.exceptions.Timeout:
            return "LLM Error: Request timed out."
        except requests.exceptions.ConnectionError:
            return "LLM Error: Cannot connect to Ollama. Ensure Ollama is running."
        except Exception as e:
            return f"LLM Error: {str(e)}"
