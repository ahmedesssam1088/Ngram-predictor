import logging

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model, normalizer, top_k=3):
        """
        Args:
            model: The trained NGramModel instance.
            normalizer: The Normalizer instance.
            top_k: Number of suggestions to return.
        """
        self.model = model
        self.normalizer = normalizer
        self.top_k = top_k

    def predict(self, user_input, require_space=False):
        """
        Processes input and returns top-k predictions.
        
        Args:
            user_input (str): The text typed by the user.
            require_space (bool): If True, only predicts if the input ends with a space.
        """
        # Step 1: Handle the "Space Trigger" for Web UI
        if require_space and not user_input.endswith(" "):
            return []

        # Step 2: Clean and tokenize the input
        # Note: We strip whitespace for the tokenizer, but we checked the original
        # user_input for the trailing space above.
        clean_input = self.normalizer.normalize(user_input)
        words = self.normalizer.word_tokenize(clean_input)
        
        if not words:
            return []

        # Step 3: Map words to <UNK> if they aren't in the vocabulary
        processed_words = [w if w in self.model.vocab else "<UNK>" for w in words]
        
        # Step 4: Get context (N-1 words)
        context_words = processed_words[-(self.model.n - 1):]
        
        # Step 5: Stupid Backoff Logic
        # Try full context, then 1 word less, then 2 words less...
        for i in range(len(context_words) + 1):
            current_context = " ".join(context_words[i:])
            
            if current_context in self.model.model_data:
                candidates = self.model.model_data[current_context]
                # Sort by highest probability
                sorted_res = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                return [word for word, prob in sorted_res[:self.top_k]]
        
        return []