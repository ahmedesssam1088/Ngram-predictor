import os
import argparse
import logging
from dotenv import load_dotenv

# Import our Module classes
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

# 1. Load configuration from config/.env
load_dotenv("config/.env")

# 2. Configure Logging (The professional version of print())
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_dataprep():
    """Executes the Data Preparation pipeline."""
    logger.info("Starting Data Preparation...")
    
    # Grab file paths from our .env file
    raw_dir = os.getenv("TRAIN_RAW_DIR")
    out_file = os.getenv("TRAIN_TOKENS")
    
    if not raw_dir or not out_file:
        logger.error("Missing environment variables. Check your config/.env file.")
        return

    norm = Normalizer()
    
    # Load and clean
    logger.info(f"Loading raw text from {raw_dir}")
    raw_text = norm.load(raw_dir)
    
    #logger.info("Stripping Gutenberg headers and cleaning text...")
    #text_no_headers = norm.strip_gutenberg(raw_text)
    text_no_headers = raw_text  # Using the new cleaning method instead of just stripping headers
    
    # Tokenize
    logger.info("Splitting text into sentences (this may take a moment)...")
    sentences = norm.sentence_tokenize(text_no_headers)
    
    # Process all sentences
    processed_sentences = []
    for sent in sentences:
        cleaned_sent = norm.normalize(sent)
        if cleaned_sent.strip():
            words = norm.word_tokenize(cleaned_sent)
            processed_sentences.append(words)
            
    # Save
    logger.info(f"Saving {len(processed_sentences)} tokenized sentences to {out_file}")
    norm.save(processed_sentences, out_file)
    logger.info("Data Preparation complete!")
def run_model():
    logger.info("Starting Model Training...")

    # Get settings from .env
    token_file = os.getenv("TRAIN_TOKENS")
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")
    n_order = int(os.getenv("NGRAM_ORDER", 3))
    unk_limit = int(os.getenv("UNK_THRESHOLD", 3))

    model = NGramModel(n=n_order, unk_threshold=unk_limit)

    model.build_vocab(token_file)
    model.train(token_file)
    model.save(model_path, vocab_path)

    logger.info("Training complete!")
def run_inference():
    logger.info("Loading model for inference...")

    # 1. Setup classes
    from src.model.ngram_model import NGramModel
    norm = Normalizer()
    model = NGramModel(n=int(os.getenv("NGRAM_ORDER", 4)))

    # 2. Load the data we saved in Step 2
    model.load(os.getenv("MODEL"), os.getenv("VOCAB"))

    # 3. Setup Predictor
    predictor = Predictor(model, norm, top_k=int(os.getenv("TOP_K", 3)))

    print("\n--- Sherlock Holmes Predictor ---")
    print("Type a phrase and hit Enter. Type 'quit' to stop.\n")

    while True:
        user_text = input("Enter text: ")
        if user_text.lower() == 'quit':
            break

        results = predictor.predict(user_text)
        print(f"Suggestions: {results}\n")
def main():
    # 3. Set up the Command Line Interface (CLI)
    parser = argparse.ArgumentParser(description="N-Gram Next-Word Predictor CLI")
    
    # Create a --step argument so we can tell the script what to do
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["dataprep", "model", "inference", "all"],
        help="Which pipeline step to run"
    )
    
    args = parser.parse_args()
    
    # 4. Route the command to the correct function
    if args.step == "dataprep":
        run_dataprep()
    elif args.step == "model":
        run_model()
    elif args.step == "inference":
        run_inference()
    elif args.step == "all":
        logger.warning("Full pipeline not yet implemented.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()