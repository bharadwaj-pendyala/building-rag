from typing import List, Dict, Any
from transformers import pipeline
import openai
import google.generativeai as genai # type: ignore


class Generator:
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self._setup_model()

    def _setup_model(self):
        if self.model_name in ['gpt-3.5-turbo', 'gpt-4']:
            openai.api_key = self.api_key
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
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def _generate_gemini(self, prompt: str, max_tokens: int = 100) -> str:
        response = self.model.generate_content(prompt, max_output_tokens=max_tokens)
        return response.text

    def _generate_huggingface(self, prompt: str, max_length: int = 100) -> str:
        generated = self.model(prompt, max_length=max_length, num_return_sequences=1)
        return generated[0]['generated_text']

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        # This method will be replaced during _setup_model
        pass
