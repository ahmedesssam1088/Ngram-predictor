import logging

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model, normalizer, top_k=3):
        self.model = model
        self.normalizer = normalizer
        self.top_k = top_k

    def predict(self, user_input, require_space=False):
        # Step 1: Handle the "Space Trigger" for Web UI
        if require_space and not user_input.endswith(" "):
            return []

        # Step 2: Clean and tokenize the input
        clean_input = self.normalizer.normalize(user_input)
        words = self.normalizer.word_tokenize(clean_input)

        if not words:
            return []

        # Step 3: Map OOV words to <UNK>
        processed_words = [w if w in self.model.vocab else "<UNK>" for w in words]

        # Step 4: Get context (N-1 words)
        context_words = processed_words[-(self.model.n - 1):]

        # Step 5: Stupid Backoff — try highest order first, down to unigram
        for i in range(len(context_words) + 1):
            current_context = " ".join(context_words[i:])
            order = len(context_words[i:]) + 1  # context length + 1 for the target word
            gram_key = f"{order}gram"

            if order == 1:
                # Unigram: model_data["1gram"] = {word: prob}, no context key needed
                candidates = self.model.model_data.get("1gram", {})
            else:
                # N-gram: model_data["Ngram"] = {context: {word: prob}}
                gram_dict = self.model.model_data.get(gram_key, {})
                candidates = gram_dict.get(current_context, {})

            if candidates:
                sorted_res = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                return [word for word, prob in sorted_res[:self.top_k]]

        return []