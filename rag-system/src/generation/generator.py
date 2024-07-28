import os
from typing import List, Dict, Any
from transformers import pipeline
from openai import OpenAI
import google.generativeai as genai

class Generator:
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self._setup_model()

    def _setup_model(self):
        if self.model_name in ['gpt-3.5-turbo', 'gpt-4']:
            self.client = OpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY")
            )
            self.generate = self._generate_openai
        elif self.model_name == 'gemini-pro':
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.generate = self._generate_gemini
        elif self.model_name in ['gpt2', 'distilgpt2']:
            self.model = pipeline('text-generation', model=self.model_name)
            self.generate = self._generate_huggingface
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _generate_openai(self, prompt: str, max_tokens: int = 100) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _generate_gemini(self, prompt: str, max_tokens: int = 100) -> str:
        response = self.model.generate_content(prompt, max_output_tokens=max_tokens)
        return response.text

    def _generate_huggingface(self, prompt: str, max_tokens: int = 100) -> str:
        generated = self.model(prompt, max_length=max_tokens, num_return_sequences=1)
        return generated[0]["generated_text"]
