import json
import os
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class NGramModel:
    def __init__(self, n=3, unk_threshold=3):
        self.n = n
        self.unk_threshold = unk_threshold
        self.vocab = set()
        # This dictionary will store our counts/probabilities
        self.model_data = {} 

    def build_vocab(self, token_file):
        """Identifies words that appear often enough to keep."""
        logger.info("Building vocabulary...")
        counts = Counter()
        
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                counts.update(line.strip().split())
        
        # Only keep words that appear more than unk_threshold times
        self.vocab = {word for word, count in counts.items() if count >= self.unk_threshold}
        self.vocab.add("<UNK>") # For words we don't recognize
        logger.info(f"Vocab size: {len(self.vocab)}")

    def train(self, token_file):
        """Counts word sequences and calculates probabilities."""
        logger.info(f"Training {self.n}-gram model...")
        
        # We store counts in a nested dictionary
        # Example: { "sherlock": {"holmes": 50, "was": 10} }
        ngram_counts = defaultdict(Counter)

        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Replace rare words with <UNK>
                words = [w if w in self.vocab else "<UNK>" for w in line.strip().split()]
                
                # Slide across the sentence to get n-grams
                for i in range(len(words)):
                    for order in range(1, self.n + 1):
                        if i + order > len(words):
                            break
                        
                        sequence = words[i : i + order]
                        context = " ".join(sequence[:-1])
                        target = sequence[-1]
                        
                        # Store how many times 'target' follows 'context'
                        ngram_counts[context][target] += 1

        # Convert counts to probabilities (MLE)
        for context, targets in ngram_counts.items():
            total_count = sum(targets.values())
            self.model_data[context] = {
                word: count / total_count for word, count in targets.items()
            }

    def save(self, model_path, vocab_path):
        """Saves the learned data to JSON files."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'w') as f:
            json.dump(self.model_data, f)
        with open(vocab_path, 'w') as f:
            json.dump(list(self.vocab), f)
        logger.info("Model and Vocab saved successfully.")

    def load(self, model_path, vocab_path):
        """Loads a previously trained model."""
        with open(model_path, 'r') as f:
            self.model_data = json.load(f)
        with open(vocab_path, 'r') as f:
            self.vocab = set(json.load(f))
        logger.info("Model loaded from disk.")