# CI/CD Workflow Documentation

## Overview
This document provides a detailed explanation of the CI/CD process for the ASI project, including setup, testing, and deployment steps. It also includes instructions on how to utilize the ASI system effectively.

## CI/CD Pipeline Configuration
The CI/CD pipeline is configured using GitHub Actions. The workflow file `.github/workflows/asi_ci_cd.yml` defines the steps involved in the CI/CD process.

### Workflow File
The workflow file includes the following steps:
1. **Setup Python**: Configures the Python environment.
2. **Install Dependencies**: Installs the required dependencies listed in `requirements.txt`.
3. **Run Tests**: Executes the unit tests to ensure code quality and functionality.
4. **Deploy**: Deploys the application if all tests pass successfully.

### Node.js Compatibility
The workflow file has been updated to use Node.js 20 compatible versions of `actions/checkout` and `actions/setup-python` to ensure compatibility with the latest Node.js versions.

## Setup Instructions
To set up the CI/CD pipeline for the ASI project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VishwamAI/ASI-Experiments-2.git
   cd ASI-Experiments-2
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Tests Locally**:
   ```bash
   python -m unittest discover tests
   ```

## Utilizing the ASI System
The ASI system consists of several modules, each with specific functionalities. Below are instructions on how to use the key modules.

### Text Generation Module
The `TextGeneration` class in `src/text_generation.py` provides text generation capabilities using a fine-tuned GPT-2 model.

#### Example Usage
```python
from src.text_generation import TextGeneration

text_gen = TextGeneration()
prompt = "Once upon a time"
generated_texts = text_gen.generate_text(prompt, max_length=50, num_return_sequences=1, num_beams=3)

for i, text in enumerate(generated_texts):
    print(f"Generated Text {i+1}: {text}")
```

### Data Analysis
The `process_few_shot_data.py` script in `SciGen/dataset/train/few-shot/` converts JSON data into a pandas DataFrame and generates a histogram plot of paper lengths.

#### Example Usage
```bash
python SciGen/dataset/train/few-shot/process_few_shot_data.py
```

The `analyze_results.py` script in `deep-abstract-generator/results/` analyzes the results from the fine-tuned GPT-2 model.

#### Example Usage
```bash
python deep-abstract-generator/results/analyze_results.py
```

## Conclusion
This document provides an overview of the CI/CD process and instructions on how to utilize the ASI system. For further details, refer to the project documentation and code comments.

## Documentation in PDF Format
To convert this markdown document to PDF format, use the following command:
```bash
pandoc CI_CD_WORKFLOW.md -o CI_CD_WORKFLOW.pdf
```
