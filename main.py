import os
import argparse
import logging
from dotenv import load_dotenv
import json

# Imports from our modules
#from inference import predictor
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from src.evaluation.evaluator import Evaluator

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
    model.build_counts_and_probabilities(token_file)
    model.save_model(os.getenv("MODEL"))
    model.save_vocab(os.getenv("VOCAB"))
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
    #predictor = Predictor(model, norm, top_k=int(os.getenv("TOP_K", 3)))
    predictor = Predictor(model, norm)

    
    print("\n--- CLI Inference Mode ---")
    print("Type your phrase and hit Enter. Type 'quit' to exit.")
    while True:
        user_text = input("\n> ")
        if user_text.lower() == "quit":
            break
        
        # In CLI, we set require_space=False because the user hit Enter anyway
        #results = predictor.predict(user_text, require_space=False)
        # Note: Predictor takes k as a parameter now
        results = predictor.predict_next(user_text, k=int(os.getenv("TOP_K", 3)))
        print(f"Suggestions: {results}")    

# def run_evaluation():
#     # 1. Setup
#     norm = Normalizer()
#     model = NGramModel(n=int(os.getenv("NGRAM_ORDER", 4)))
#     model.load(os.getenv("MODEL"), os.getenv("VOCAB"))

#     # 2. Process the Evaluation Book first (just like training data)
#     # Note: You should have 'The Valley of Fear' in data/raw/eval/ [cite: 14]
#     eval_raw = os.getenv("EVAL_RAW_DIR")
#     eval_tokens = os.getenv("EVAL_TOKENS")

#     # Use Normalizer to prep the eval data
#     raw_text = norm.load(eval_raw)
#     clean_text = norm.strip_gutenberg(raw_text)
#     sentences = norm.sentence_tokenize(clean_text)

# # Ensure this part is exactly like this:
#     processed = []
#     for s in sentences:
#         cleaned = norm.normalize(s)
#         tokens = norm.word_tokenize(cleaned)
#         if tokens:
#             processed.append(tokens)
    
#     # CRITICAL: This saves the tokens so evaluator.py can read them
#     norm.save(processed, eval_tokens)
#     # 3. Run Evaluator
#     evaluator = Evaluator(model, norm)
#     evaluator.run(eval_tokens)
def run_evaluation():
    norm  = Normalizer()
    model = NGramModel(n=int(os.getenv("NGRAM_ORDER", 4)))
    model.load(os.getenv("MODEL"), os.getenv("VOCAB"))

    eval_raw    = os.getenv("EVAL_RAW_DIR")
    eval_tokens = os.getenv("EVAL_TOKENS")

    raw_text   = norm.load(eval_raw)
    clean_text = norm.strip_gutenberg(raw_text)
    sentences  = norm.sentence_tokenize(clean_text)

    processed = []
    for s in sentences:
        cleaned = norm.normalize(s)
        tokens  = norm.word_tokenize(cleaned)
        if tokens:
            processed.append(tokens)

    # Save as plain text: one sentence per line, tokens space-separated
    norm.save(processed, eval_tokens)
    evaluator = Evaluator(model, norm)
    evaluator.run(eval_tokens)

def main():
    parser = argparse.ArgumentParser(description="N-Gram Predictor")
    parser.add_argument("--step", choices=["dataprep", "model", "inference", "evaluate", "all"])
    args = parser.parse_args()

    if args.step == "dataprep":
        run_dataprep()
    elif args.step == "model":
        run_model()
    elif args.step == "inference":
        run_inference()
    elif args.step == "evaluate":
        run_evaluation()
    elif args.step == "all":
        run_dataprep()
        run_model()
        run_inference()
        run_evaluation()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()