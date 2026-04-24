import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class LLMGenerator:
    def __init__(self, model_name="gemini-flash-lite-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("[WARNING] GOOGLE_API_KEY not found in environment variables.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, query, context):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            return "LLM Error: Please provide a valid GOOGLE_API_KEY in your .env file. Visit https://aistudio.google.com/app/apikey to get one."

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
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return "LLM Error: No response received from Gemini."

            return response.text.strip()

        except Exception as e:
            return f"LLM Error: {str(e)}"

