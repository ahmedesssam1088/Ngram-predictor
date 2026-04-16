# 🕵️‍♂️ Sherlock N-Gram Predictor

The Sherlock N-Gram Predictor is a statistical language model designed to predict the next word in a sequence by learning the linguistic patterns of Arthur Conan Doyle's Sherlock Holmes novels. Built using a 4-gram Markov chain approach with Maximum Likelihood Estimation (MLE), the system processes raw text, handles unseen words via a frequency-based vocabulary threshold, and utilizes "Stupid Backoff" logic to provide reliable suggestions even when specific high-order contexts have not been previously encountered.

## 🛠️ Requirements
* **Python Version**: 3.11+
* **Dependencies**: All necessary libraries are listed in `requirements.txt`. Install them using the command provided in the Setup section.

## ⚙️ Setup

1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd ngram-predictor
    ```

2.  **Create and Activate Anaconda Environment**:
    ```bash
    conda create -n ngram-env python=3.11
    conda activate ngram-env
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Populate Configuration**:
    Create a file at `config/.env` and populate it with your specific project settings (N-gram order, thresholds, and file paths).

5.  **Download Raw Data**:
    Ensure the raw `.txt` files for the Sherlock Holmes novels are placed in the following folders:
    * Training files (Adventures, Memoirs, Return, Hound) -> `data/raw/train/`
    * Evaluation file (Valley of Fear) -> `data/raw/eval/`

## 🚀 Usage

Follow these steps in sequence to process the data, train the model, and interact with the results:

1.  **Data Preparation**: Clean and tokenize the raw text.
    ```bash
    python main.py --step dataprep
    ```

2.  **Model Training**: Build the vocabulary and probability tables.
    ```bash
    python main.py --step model
    ```
3.  **Interactive CLI**: Start the interactive CLI prediction loop.
    ```bash
    python main.py --step inference
    ```

4.  **Model Evaluation**: Calculate perplexity on the unseen evaluation set.
    ```bash
    python main.py --step evaluate
    ```

5.  **Interactive UI**: Launch the Streamlit-based "Smart Keyboard" interface.
    ```bash
    streamlit run src/ui/app.py
    ```

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