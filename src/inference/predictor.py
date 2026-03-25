class Predictor:
    def __init__(self, model, normalizer, top_k=3):
        # We "inject" the model and normalizer here
        self.model = model
        self.normalizer = normalizer
        self.top_k = top_k

    def predict(self, user_input):
        # 1. Normalize and Tokenize
        clean_input = self.normalizer.normalize(user_input)
        words = self.normalizer.word_tokenize(clean_input)
        
        # 2. Convert words to OOV (Out of Vocabulary) if they aren't in our vocab
        processed_words = [w if w in self.model.vocab else "<UNK>" for w in words]
        
        # 3. Get context (up to N-1 words)
        context_words = processed_words[-(self.model.n - 1):]
        
        # 4. Stupid Backoff Logic
        for i in range(len(context_words) + 1):
            current_context = " ".join(context_words[i:])
            if current_context in self.model.model_data:
                candidates = self.model.model_data[current_context]
                # Sort by probability
                sorted_res = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                return [word for word, prob in sorted_res[:self.top_k]]
        
        return [] # Return empty if nothing found