import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class NGramModel:
    def __init__(self, n=4, unk_threshold=3):
        self.n = n
        self.unk_threshold = unk_threshold
        self.vocab = set()
        # Initialize model_data with keys for each order
        self.model_data = {f"{i}-gram": {} for i in range(1, n + 1)}

    def build_vocab(self, token_file):
        """Build vocabulary; apply UNK_THRESHOLD from .env"""
        counts = defaultdict(int)
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                for token in tokens:
                    counts[token] += 1
                    
        self.vocab = {word for word, count in counts.items() if count >= self.unk_threshold}
        self.vocab.add("<UNK>")

    def build_counts_and_probabilities(self, token_file):
        """Count all n-grams and compute MLE probabilities structured by order."""
        # Temporary nested dictionary to hold counts: {order: {context: {word: count}}}
        all_counts = {i: defaultdict(lambda: defaultdict(int)) for i in range(1, self.n + 1)}
        
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                
                tokens = [t if t in self.vocab else "<UNK>" for t in tokens]
                
                for order in range(1, self.n + 1):
                    for i in range(len(tokens) - order + 1):
                        ngram = tokens[i:i+order]
                        context = " ".join(ngram[:-1])
                        word = ngram[-1]
                        all_counts[order][context][word] += 1

        # Compute MLE Probabilities into the structured model_data
        for order in range(1, self.n + 1):
            order_key = f"{order}-gram"
            for context, words in all_counts[order].items():
                total_context_count = sum(words.values())
                self.model_data[order_key][context] = {
                    word: count/total_context_count 
                    for word, count in words.items()
                }

    def lookup(self, context):
        """Backoff lookup: Searches structured model_data from highest order to 1-gram."""
        for order in range(self.n, 0, -1):
            order_key = f"{order}-gram"
            current_context = " ".join(context[-(order-1):]) if order > 1 else ""
            
            if current_context in self.model_data[order_key]:
                return self.model_data[order_key][current_context]
        return {}

    def save_model(self, model_path):
        """Save structured probability tables to model.json"""
        with open(model_path, 'w', encoding='utf-8') as f:
            # indent=4 makes the JSON readable so you can verify the 1-gram, 2-gram structure
            json.dump(self.model_data, f, indent=4)

    def save_vocab(self, vocab_path):
        """Save vocabulary list to vocab.json"""
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.vocab), f, indent=4)

    def load(self, model_path, vocab_path):
        """Load structured model.json and vocab.json."""
        with open(model_path, 'r', encoding='utf-8') as f:
            self.model_data = json.load(f)
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = set(json.load(f))