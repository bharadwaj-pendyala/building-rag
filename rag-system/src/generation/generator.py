from transformers import pipeline


class Generator:
    def __init__(self):
        self.model = pipeline("text-generation", model="gpt2")

    def generate(self, prompt, max_length=100):
        generated = self.model(prompt, max_length=max_length, num_return_sequences=1)
        return generated[0]["generated_text"]
