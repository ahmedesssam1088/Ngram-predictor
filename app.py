import streamlit as st
import os
import sys
from dotenv import load_dotenv
from st_keyup import st_keyup  # The "Magic" component for live typing

# Path fix for src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

# 1. Configuration & Load Env
st.set_page_config(page_title="Sherlock Instant", page_icon="🕵️‍♂️")
load_dotenv("config/.env")

# 2. Resource Loader (Cached)
@st.cache_resource
def load_engine():
    try:
        norm = Normalizer()
        n_order = int(os.getenv("NGRAM_ORDER", 4))
        model = NGramModel(n=n_order)
        
        model_path = os.getenv("MODEL")
        vocab_path = os.getenv("VOCAB")
        
        if not os.path.exists(model_path):
            return None
            
        model.load(model_path, vocab_path)
        return Predictor(model, norm, top_k=int(os.getenv("TOP_K", 3)))
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None

predictor = load_engine()

# 3. User Interface
st.title("🕵️‍♂️ Sherlock Holmes Instant Predictor")
st.markdown("Type and watch the suggestions update **instantly** without hitting Enter.")

if predictor is None:
    st.warning("⚠️ Model files not found. Run `python main.py --step model` first.")
else:
    # 4. The Key-Up Input (TRUE REAL-TIME)
    user_input = st_keyup(
        "Start typing your sentence:", 
        key="live_input", 
        placeholder="The adventure of "
    )

    # 5. Prediction Logic
    if user_input:
        # We only predict if there is a trailing space (user finished a word)
        if user_input.endswith(" "):
            # Call predictor (require_space=False because we checked it here)
            suggestions = predictor.predict(user_input, require_space=False)
            
            if suggestions:
                st.write("### Next Word Suggestions:")
                cols = st.columns(len(suggestions))
                for i, word in enumerate(suggestions):
                    with cols[i]:
                        # Using buttons as visual suggestions
                        st.button(word, key=f"btn_{word}_{i}")
            else:
                st.caption("_No specific match found in the Sherlock corpus._")
        else:
            # Hint for the user while they are typing a word
            st.caption("Press **Space** to see the next word...")

# Sidebar for metadata
with st.sidebar:
    st.header("Technical Specs")
    st.write(f"**N-Gram Order:** {os.getenv('NGRAM_ORDER')}")
    st.write(f"**Top-K Results:** {os.getenv('TOP_K')}")
    if st.button("Clear App Cache"):
        st.cache_resource.clear()
        st.rerun()