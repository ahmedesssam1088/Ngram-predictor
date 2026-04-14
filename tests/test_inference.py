import pytest
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer

@pytest.fixture
def predictor():
    # Setup a tiny mock model
    model = NGramModel(n=2)
    model.vocab = {"the", "adventure"}
    model.model_data = {"1-gram": {"the": 0.5, "adventure": 0.5}, "2-gram": {"the": {"adventure": 1.0}}}
    return Predictor(model, Normalizer())

def test_predict_next(predictor):
    # Exactly k predictions
    res = predictor.predict_next("the", k=1)
    assert len(res) == 1
    
    # Sorted by probability
    # In our mock, 'the' -> 'adventure' is 1.0 prob
    res = predictor.predict_next("the", k=2)
    assert res[0] == "adventure"

def test_map_oov(predictor):
    context = ["the", "mystery"]
    safe = predictor.map_oov(context)
    assert safe == ["the", "<UNK>"]