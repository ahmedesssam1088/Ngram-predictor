# 🕵️‍♂️ Sherlock N-Gram Predictor

A statistical language model built from scratch in Python to predict the next word in a sequence, specifically trained on the **Sherlock Holmes** canon. This project demonstrates a complete NLP pipeline: from raw text processing to Maximum Likelihood Estimation (MLE) probability modeling and real-time inference via a Streamlit UI.

## 🚀 Milestone Achievements

### 1. Advanced Data Normalization
* **Gutenberg Stripping**: Custom regex logic to automatically identify and remove legal headers/footers from Project Gutenberg ebooks.
* **Robust Tokenization**: Precise handling of punctuation, casing, and numerical data to ensure high-quality training tokens.

### 2. Hierarchical N-Gram Model
* **Structured Storage**: Implemented a multi-level JSON structure (`1-gram` through `4-gram`) for clear model inspection.
* **Stupid Backoff Logic**: A dedicated `lookup()` method that serves as the single source of truth for backoff logic, falling back through n-gram orders until a match is found.
* **Vocabulary Management**: Integrated `<UNK>` token handling based on frequency thresholds to manage Out-of-Vocabulary (OOV) words.

### 3. Interactive Inference UI
* **Smart Keyboard**: Developed a Streamlit interface using `st-keyup` for instant word suggestions as the user types.
* **OOV Mapping**: The predictor dynamically maps unknown user inputs to `<UNK>` to maintain stability during real-time sessions.

### 4. Comprehensive Evaluation
* **Perplexity Analysis**: Verified model "surprise" on unseen text (*The Valley of Fear*).
* **Performance Metrics**: Achieved a perplexity of **~20.99**, successfully narrowing down language choices to ~21 possibilities per word.

### 5. Verified Engineering
* **Unit Testing**: 100% pass rate across **9 unit tests** using `pytest`, covering normalization, backoff logic, probability summation, and inference.

## 🛠️ Installation & Setup

1.  **Environment**:
    ```bash
    conda create -n ngram-env python=3.11
    conda activate ngram-env
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    Adjust variables in `config/.env` to tune `NGRAM_ORDER` or `UNK_THRESHOLD`.

## 📖 Execution Guide

| Goal | Command |
| :--- | :--- |
| **Prepare Data** | `python main.py --step dataprep` |
| **Train Model** | `python main.py --step model` |
| **Run Predication** | `python main.py --step inference` |
| **Run Evaluation** | `python main.py --step evaluate` |
| **Launch UI** | `streamlit run src/ui/app.py` |
| **Run Tests** | `python -m pytest tests/` |

## 📂 Project Structure
```text
ngram-predictor/
├── config/
│   └── .env                    # Environment variables
├── data/
│   ├── model/                  # Where trained weights live
│   │   ├── model.json
│   │   └── vocab.json
│   ├── processed/              # Normalized token files
│   │   ├── eval_tokens.txt
│   │   └── train_tokens.txt
│   └── raw/                    # Original text files
│       ├── eval/
│       │   └── 3289-0.txt      (The Valley of Fear)
│       └── train/
│           ├── adventures.txt
│           ├── hound.txt
│           ├── memoirs.txt
│           └── return.txt
├── src/
│   ├── __init__.py             # Makes src a package
│   ├── data_prep/
│   │   ├── __init__.py
│   │   └── normalizer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── ngram_model.py
│   └── ui/
│       ├── __init__.py
│       └── app.py
├── tests/
│   ├── __init__.py             # Recommended for pytest
│   ├── test_data_prep.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   └── test_model.py
├── .gitignore                  # Should ignore __pycache__ and .pytest_cache
├── conftest.py                 # Empty file to help pytest find 'src'
├── download_data.py
├── main.py                     # CLI entry point
├── README.md
└── requirements.txt