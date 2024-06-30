import pandas as pd

def analyze_results(file_path, chunk_size=1000):
    # Read the CSV file in chunks
    chunks = pd.read_csv(file_path, chunksize=chunk_size)

    # Process the first chunk
    first_chunk = next(chunks)

    # Print the first few rows of the first chunk
    print(first_chunk.head())

if __name__ == "__main__":
    file_path = "results_from_fine-tuned-gpt2_bu.csv"
    analyze_results(file_path)
