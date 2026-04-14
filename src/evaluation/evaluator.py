import math
import logging
import json

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Computes perplexity on a held-out evaluation corpus to measure model performance.
    """

    def __init__(self, model, normalizer):
        """Accepts a pre-loaded NGramModel and Normalizer instance."""
        self.model = model
        self.normalizer = normalizer

    def score_word(self, word, context):
            """
            Return log2 P(word | context) via NGramModel.lookup(). [cite: 173]
            """
            # Ensure context is a list for the lookup method
            predictions = self.model.lookup(context) 
            
            if word in predictions:
                prob = predictions[word]
                # Use log2 for entropy calculations [cite: 172]
                return math.log2(prob)
            return None

    def compute_perplexity(self, eval_file):
        total_log_prob = 0
        word_count = 0
        skipped_count = 0

        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip().split() for line in f if line.strip()]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load eval file: {e}")
            return 0, 0, 0

        for words in sentences:
            if not words:
                continue

            # Map OOV words to <UNK> before scoring
            words = [w if w in self.model.vocab else "<UNK>" for w in words]

            for i in range(len(words)):
                word = words[i]
                context = words[max(0, i - (self.model.n - 1)):i]

                log_p = self.score_word(word, context)

                if log_p is not None:
                    total_log_prob += log_p
                    word_count += 1
                else:
                    skipped_count += 1

        if word_count == 0:
            return 0, 0, 0

        cross_entropy = -(total_log_prob / word_count)
        perplexity = math.pow(2, cross_entropy)
        return perplexity, word_count, skipped_count 
 
    def run(self, eval_file):
        """Orchestrates compute_perplexity and prints results."""
        logger.info(f"Evaluating model on {eval_file}...")
        perp, total, skipped = self.compute_perplexity(eval_file)
        
        print(f"\n--- Model Evaluation Results ---")
        print(f"Perplexity: {perp:.2f}") # 
        print(f"Words evaluated: {total}") # 
        print(f"Words skipped (zero probability): {skipped}") #