import pytest
from src.data_prep.normalizer import Normalizer

@pytest.fixture
def norm():
    return Normalizer()

def test_normalize_steps(norm):
    # Test lowercase
    assert norm.normalize("SHERLOCK") == "sherlock"
    # Test punctuation
    assert norm.normalize("holmes!") == "holmes"
    # Test numbers
    assert norm.normalize("room 221b") == "room b"
    # Test whitespace
    assert norm.normalize("  word  ") == "word"
    # Sequence test
    assert norm.normalize("1. Hello, World!") == "hello world"

def test_strip_gutenberg(norm):
    # This string matches your regex: *** START OF THE PROJECT GUTENBERG EBOOK ... ***
    text = (
        "*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES ***\n"
        "Real Content\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES ***"
    )
    
    stripped = norm.strip_gutenberg(text)
    
    # 1. Verify the core content is kept
    assert "Real Content" in stripped
    
    # 2. Verify the markers themselves are removed
    assert "START OF" not in stripped
    assert "PROJECT GUTENBERG EBOOK" not in stripped
    
    # 3. Test the 'No Marker' fallback case
    no_marker_text = "Just a normal string of text."
    assert norm.strip_gutenberg(no_marker_text) == no_marker_text

def test_tokenizers(norm):
    text = "This is a sentence. This is another."
    sentences = norm.sentence_tokenize(text)
    assert len(sentences) >= 1
    
    tokens = norm.word_tokenize("word1 word2")
    assert tokens == ["word1", "word2"]
    assert "" not in tokens