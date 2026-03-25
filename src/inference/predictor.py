import logging

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model, normalizer, top_k=3):
        """
        Args:
            model: The trained NGramModel instance.
            normalizer: The Normalizer instance to clean input text.
            top_k: How many suggestions to return.
        """
        self.model = model
        self.normalizer = normalizer
        self.top_k = top_k

    def predict(self, user_input):
        """Processes input and returns top-k next word predictions."""
        # 1. Clean the user's input just like we cleaned the books
        clean_input = self.normalizer.normalize(user_input)
        words = self.normalizer.word_tokenize(clean_input)
        
        # 2. Get the last (N-1) words as context
        # If N=4, we look at the last 3 words.
        context_words = words[-(self.model.n - 1):]
        
        # 3. Try to find predictions (Backoff Logic)
        predictions = []
        
        # Start with the full context, then keep removing the first word (Backing off)
        for i in range(len(context_words) + 1):
            current_context = " ".join(context_words[i:])
            
            if current_context in self.model.model_data:
                # Get the words and their probabilities
                candidates = self.model.model_data[current_context]
                # Sort them by highest probability
                sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                predictions = [word for word, prob in sorted_candidates[:self.top_k]]
                break # We found a match, so we stop "backing off"
        
        return predictions