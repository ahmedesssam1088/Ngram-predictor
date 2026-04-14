import streamlit as st
import os
import sys
from dotenv import load_dotenv
from st_keyup import st_keyup

# Path fix
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

# 1. Config & Load
st.set_page_config(page_title="N-Gram Predictor", page_icon="🕵️‍♂️")
load_dotenv("config/.env")

# 2. Initialize Session State
if "current_text" not in st.session_state:
    st.session_state.current_text = ""
if "input_key" not in st.session_state:
    st.session_state.input_key = 0  # We use this to force-refresh the text box

# 3. Resource Loader
@st.cache_resource
def load_engine():
    norm = Normalizer()
    model = NGramModel(n=int(os.getenv("NGRAM_ORDER", 4)))
    model_path = os.getenv("MODEL")
    vocab_path = os.getenv("VOCAB")
    if os.path.exists(model_path):
        model.load(model_path, vocab_path)
        return Predictor(model, norm, top_k=int(os.getenv("TOP_K", 3)))
    return None

predictor = load_engine()

# --- Callback Function ---
def handle_click(word):
    # 1. Update the text
    st.session_state.current_text += f"{word} "
    # 2. Increment the key to force the text box to update its UI
    st.session_state.input_key += 1

# 4. UI Layout
st.title("🕵️‍♂️ N-Gram Next-Word Predictor")

if predictor is None:
    st.error("Model not found. Please train it first.")
else:
    # 5. The Input Box with Dynamic Key
    # Changing the 'key' whenever a button is pressed forces a redraw
    user_input = st_keyup(
        "Type or click suggestions:", 
        value=st.session_state.current_text,
        key=f"live_input_{st.session_state.input_key}",
        placeholder="The adventure of "
    )
    
    # Keep the state in sync if the user types manually
    st.session_state.current_text = user_input

    # 6. Prediction Logic
    if user_input:
        if user_input.endswith(" "):
            suggestions = predictor.predict(user_input, require_space=False)
            
            if suggestions:
                st.write("### Suggestions:")
                cols = st.columns(len(suggestions))
                for i, word in enumerate(suggestions):
                    with cols[i]:
                        # Use the callback function
                        st.button(
                            word, 
                            key=f"btn_{word}_{i}", 
                            on_click=handle_click, 
                            args=(word,)
                        )
            else:
                st.caption("_No predictions found._")
        else:
            st.caption("Press **Space** to see predictions...")

# Sidebar
with st.sidebar:
    if st.button("Clear Everything"):
        st.session_state.current_text = ""
        st.session_state.input_key += 1
        st.rerun()