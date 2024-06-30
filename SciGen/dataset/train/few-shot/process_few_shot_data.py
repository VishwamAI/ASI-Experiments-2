import json
import pandas as pd
import matplotlib.pyplot as plt

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def convert_to_dataframe(data):
    records = []
    for key, value in data.items():
        record = {
            'paper': value['paper'],
            'paper_id': value['paper_id'],
            'table_caption': value['table_caption'],
            'table_column_names': value['table_column_names'],
            'table_content_values': value['table_content_values']
        }
        records.append(record)
    df = pd.DataFrame(records)
    return df

def plot_paper_lengths(df):
    paper_lengths = df.groupby('paper').size()
    plt.figure(figsize=(10, 6))
    paper_lengths.plot(kind='hist', bins=20, edgecolor='black')
    plt.title('Distribution of Paper Lengths (Number of Table Entries per Paper)')
    plt.xlabel('Number of Table Entries')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('paper_lengths_distribution.png')
    plt.show()

def main():
    file_path = 'train.json'
    data = read_json_file(file_path)
    df = convert_to_dataframe(data)
    print(df.head())
    plot_paper_lengths(df)

if __name__ == "__main__":
    main()
