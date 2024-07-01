from bert_text_generation import BERTTextGeneration

def main():
    text_gen = BERTTextGeneration()
    prompt = "Once upon a time"
    generated_text = text_gen.generate_text(prompt, max_length=50, num_return_sequences=1)
    print("Generated Text:")
    print(generated_text[0])

if __name__ == "__main__":
    main()
