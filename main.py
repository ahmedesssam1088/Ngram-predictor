import os
import argparse
import logging
from dotenv import load_dotenv

# Imports from our modules
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

# 1. Load configuration
load_dotenv("config/.env")

# 2. Configure Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_dataprep():
    logger.info("Starting Data Preparation...")
    raw_dir = os.getenv("TRAIN_RAW_DIR")
    out_file = os.getenv("TRAIN_TOKENS")
    
    norm = Normalizer()
    raw_text = norm.load(raw_dir)
    text_no_headers = norm.strip_gutenberg(raw_text)
    sentences = norm.sentence_tokenize(text_no_headers)
    
    processed_sentences = []
    for sent in sentences:
        cleaned = norm.normalize(sent)
        if cleaned.strip():
            processed_sentences.append(norm.word_tokenize(cleaned))
            
    norm.save(processed_sentences, out_file)
    logger.info(f"Saved {len(processed_sentences)} sentences to {out_file}")

def run_model():
    logger.info("Starting Model Training...")
    model = NGramModel(
        n=int(os.getenv("NGRAM_ORDER", 4)),
        unk_threshold=int(os.getenv("UNK_THRESHOLD", 3))
    )
    token_file = os.getenv("TRAIN_TOKENS")
    model.build_vocab(token_file)
    model.train(token_file)
    model.save(os.getenv("MODEL"), os.getenv("VOCAB"))
    logger.info("Model Training Complete.")

def run_inference():
    logger.info("Starting Inference Mode...")
    # Initialize
    norm = Normalizer()
    model = NGramModel(n=int(os.getenv("NGRAM_ORDER", 4)))
    
    # Load files
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")
    if not os.path.exists(model_path):
        logger.error("Model files not found. Run --step model first.")
        return
        
    model.load(model_path, vocab_path)
    predictor = Predictor(model, norm, top_k=int(os.getenv("TOP_K", 3)))
    
    print("\n--- CLI Inference Mode ---")
    print("Type your phrase and hit Enter. Type 'quit' to exit.")
    
    while True:
        user_text = input("\n> ")
        if user_text.lower() == "quit":
            break
        
        # In CLI, we set require_space=False because the user hit Enter anyway
        results = predictor.predict(user_text, require_space=False)
        print(f"Suggestions: {results}")

def main():
    parser = argparse.ArgumentParser(description="N-Gram Predictor")
    parser.add_argument("--step", choices=["dataprep", "model", "inference", "all"])
    args = parser.parse_args()

    if args.step == "dataprep":
        run_dataprep()
    elif args.step == "model":
        run_model()
    elif args.step == "inference":
        run_inference()
    elif args.step == "all":
        run_dataprep()
        run_model()
        run_inference()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()