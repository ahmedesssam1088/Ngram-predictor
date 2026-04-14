import streamlit as st
import os
import sys
from dotenv import load_dotenv
from st_keyup import st_keyup

# Go UP two levels: from src/ui -> src -> root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_path)

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
    """Load model and predictor once and cache them for performance."""
    try:
        norm = Normalizer()
        n_order = int(os.getenv("NGRAM_ORDER", 4))
        model = NGramModel(n=n_order)
        
        model_path = os.getenv("MODEL")
        vocab_path = os.getenv("VOCAB")
        
        if os.path.exists(model_path):
            model.load(model_path, vocab_path)
            # FIX: Removed top_k from __init__ to match your updated class structure
            return Predictor(model, norm) 
        return None
    except Exception as e:
        st.error(f"Error initializing model: {e}")
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
            suggestions = predictor.predict_next(user_input, k=3)
            
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
    st.header("Model Settings")
    st.write(f"**N-Gram Order:** {os.getenv('NGRAM_ORDER')}")
    st.write(f"**Vocab Threshold:** {os.getenv('UNK_THRESHOLD')}")
    
    st.divider()
    
    if st.button("Clear Everything", use_container_width=True):
        st.session_state.current_text = ""
        st.session_state.input_key += 1
        st.rerun()