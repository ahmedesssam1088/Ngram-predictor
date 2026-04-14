import logging

logger = logging.getLogger(__name__)

class Predictor:
    """
    Accepts a pre-loaded NGramModel and Normalizer via the constructor, 
    normalizing input text, and returning the top-k predicted next words 
    sorted by probability. Backoff lookup is delegated to NGramModel.lookup().
    """

    def __init__(self, model, normalizer):
        """Accept a pre-loaded NGramModel and Normalizer instance. Do not load files here."""
        self.model = model
        self.normalizer = normalizer

    def normalize(self, text):
        """Call Normalizer.normalize(text); extract last NGRAM_ORDER - 1 words as context"""
        clean_text = self.normalizer.normalize(text)
        tokens = self.normalizer.word_tokenize(clean_text)
        return tokens[-(self.model.n - 1):] if self.model.n > 1 else []

    def map_oov(self, context):
        """Replace out-of-vocabulary words with <UNK>"""
        return [w if w in self.model.vocab else "<UNK>" for w in context]

    def predict_next(self, text, k):
        """Orchestrate normalize -> map_oov -> NGramModel.lookup() -> return top-k words"""
        # 1. Get context
        context = self.normalize(text)
        
        # 2. Handle OOV
        safe_context = self.map_oov(context)
        
        # 3. Delegate Backoff to Model
        predictions = self.model.lookup(safe_context)
        
        # 4. Sort and return top-k
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [word for word, prob in sorted_predictions[:k]]