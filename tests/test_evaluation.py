import pytest
import math
from src.evaluation.evaluator import Evaluator
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer

@pytest.fixture
def trained_model(tmp_path):
    """Create a tiny dummy model for evaluation testing."""
    token_file = tmp_path / "dummy_tokens.txt"
    # Provide simple text for the model to learn
    token_file.write_text("the dog\nthe dog\nthe cat\n")
    
    model = NGramModel(n=2, unk_threshold=2)
    model.build_vocab(str(token_file))
    model.build_counts_and_probabilities(str(token_file))
    return model

def test_evaluator_scoring(trained_model):
    evaluator = Evaluator(trained_model, Normalizer())
    
    # 'the' followed by 'dog' was seen twice, so it should have a probability
    score = evaluator.score_word("dog", ["the"])
    assert isinstance(score, float)
    assert score < 0  # Log probabilities are negative
    
    # A word/context combo never seen should return None (Zero Probability)
    assert evaluator.score_word("bird", ["the"]) is None

def test_compute_perplexity(trained_model, tmp_path):
    evaluator = Evaluator(trained_model, Normalizer())
    eval_file = tmp_path / "eval.txt"
    # Use words the model knows so perplexity isn't 0
    eval_file.write_text("the dog the dog")
    
    perp, count, skipped = evaluator.compute_perplexity(str(eval_file))
    
    # Perplexity should be a valid number >= 1
    assert isinstance(perp, float)
    assert perp >= 1.0
    assert count > 0