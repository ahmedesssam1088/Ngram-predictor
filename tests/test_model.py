import pytest
from src.model.ngram_model import NGramModel

@pytest.fixture
def trained_model(tmp_path):
    # Create a tiny dummy token file
    token_file = tmp_path / "dummy_tokens.txt"
    token_file.write_text("the dog\nthe dog\nthe cat\n")
    
    model = NGramModel(n=2, unk_threshold=2)
    model.build_vocab(str(token_file))
    model.build_counts_and_probabilities(str(token_file))
    return model

def test_build_vocab(trained_model):
    assert "the" in trained_model.vocab
    assert "dog" in trained_model.vocab
    # 'cat' only appeared once, threshold is 2
    assert "cat" not in trained_model.vocab
    assert "<UNK>" in trained_model.vocab

def test_lookup_and_probabilities(trained_model):
    # Seen context
    res = trained_model.lookup(["the"])
    assert len(res) > 0
    # Sum of probabilities should be ~1
    assert sum(res.values()) == pytest.approx(1.0)
    
    # Unseen context (Backs off to unigram)
    res_unseen = trained_model.lookup(["unknown"])
    assert len(res_unseen) > 0
    
    # Fail case: Empty dict if nothing matches (even unigrams)
    # Only happens if model_data is literally empty
    empty_model = NGramModel(n=2)
    assert empty_model.lookup(["nothing"]) == {}