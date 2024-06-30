import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGeneration:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=50, num_return_sequences=1, num_beams=1):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            early_stopping=True,
            num_beams=num_beams
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

if __name__ == "__main__":
    text_gen = TextGeneration()
    prompt = "Once upon a time"
    generated_texts = text_gen.generate_text(prompt)
    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i+1}: {text}")
