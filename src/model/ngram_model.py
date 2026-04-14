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
        self.model_data = {}  # {"1gram": {...}, "2gram": {...}, ...}

    def lookup(self, context):
        """
        Backoff lookup: try highest-order context first, fall back to unigram.
        context: list of preceding words (already UNK-mapped).
        """
        for order in range(self.n, 0, -1):
            if order == 1:
                # Unigram: model_data["1gram"] = {word: prob}
                candidates = self.model_data.get("1gram", {})
                if candidates:
                    return candidates
            else:
                # N-gram: model_data["Ngram"] = {context_str: {word: prob}}
                gram_key = f"{order}gram"
                context_words = context[-(order - 1):]
                context_key = " ".join(context_words)
                candidates = self.model_data.get(gram_key, {}).get(context_key, {})
                if candidates:
                    return candidates

        return {}    

    def build_vocab(self, token_file):
        """Identifies words that appear often enough to keep."""
        logger.info("Building vocabulary...")
        counts = Counter()

        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                counts.update(line.strip().split())

        self.vocab = {word for word, count in counts.items() if count >= self.unk_threshold}
        self.vocab.add("<UNK>")
        logger.info(f"Vocab size: {len(self.vocab)}")

    def train(self, token_file):
        """Counts word sequences and calculates probabilities per order."""
        logger.info(f"Training {self.n}-gram model...")

        # Separate counts per order: order -> context -> Counter of targets
        # For unigrams, context is None (no prefix)
        order_counts = {order: defaultdict(Counter) for order in range(1, self.n + 1)}

        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = [w if w in self.vocab else "<UNK>" for w in line.strip().split()]

                for i in range(len(words)):
                    for order in range(1, self.n + 1):
                        if i + order > len(words):
                            break

                        sequence = words[i: i + order]
                        context = " ".join(sequence[:-1])  # empty string "" for unigrams
                        target = sequence[-1]

                        order_counts[order][context][target] += 1

        # Convert counts to MLE probabilities, structured by order
        for order in range(1, self.n + 1):
            key = f"{order}gram"

            if order == 1:
                # Unigram: divide each word count by total word count
                total_words = sum(
                    sum(targets.values())
                    for targets in order_counts[order].values()
                )
                self.model_data[key] = {
                    target: count / total_words
                    for targets in order_counts[order].values()
                    for target, count in targets.items()
                }
            else:
                # N-gram (n>1): divide by the count of that context
                self.model_data[key] = {}
                for context, targets in order_counts[order].items():
                    total_context = sum(targets.values())
                    self.model_data[key][context] = {
                        word: count / total_context
                        for word, count in targets.items()
                    }

        logger.info("Training complete.")

    def save(self, model_path, vocab_path):
        """Saves the learned data to JSON files."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'w') as f:
            json.dump(self.model_data, f, indent=2)
        with open(vocab_path, 'w') as f:
            json.dump(list(self.vocab), f, indent=2)
        logger.info("Model and vocab saved successfully.")

    def load(self, model_path, vocab_path):
        """Loads a previously trained model."""
        with open(model_path, 'r') as f:
            self.model_data = json.load(f)
        with open(vocab_path, 'r') as f:
            self.vocab = set(json.load(f))
        logger.info("Model loaded from disk.")