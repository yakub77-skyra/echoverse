# echoverse_app.py

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from TTS.api import TTS
import tempfile
import os

# -------------------------------
# MODEL SETUP
# -------------------------------
@st.cache_resource
def load_granite_model():
    model_path = "ibm-granite/granite-3.3-8b-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, device

model, tokenizer, device = load_granite_model()

# -------------------------------
# OPEN SOURCE TTS SETUP
# -------------------------------
@st.cache_resource
def load_tts_model():
    return TTS("tts_models/en/vctk/vits")

tts = load_tts_model()

# -------------------------------
# FUNCTION: Tone Rewrite
# -------------------------------
def rewrite_text_with_tone(original_text, tone):
    prompt = f"""
You are a professional narrator rewriting text in a {tone.lower()} tone.
Maintain the original meaning but enhance style, emotion, and readability.

Original Text:
{original_text}

Rewritten {tone} Version:
"""

    conv = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        conv,
        return_tensors="pt",
        thinking=True,
        return_dict=True,
        add_generation_prompt=True
    ).to(device)

    set_seed(42)
    output = model.generate(**input_ids, max_new_tokens=1024)
    rewritten = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return rewritten.strip()

# -------------------------------
# FUNCTION: Generate Audio
# -------------------------------
def generate_audio(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.tts_to_file(text=text, file_path=tmp.name)
        return tmp.name

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="EchoVerse - Granite Audiobook Generator", layout="wide")
st.title("🎧 EchoVerse – Generative AI Audiobook Creator (Granite-Powered)")

st.sidebar.header("Configuration")
tone = st.sidebar.selectbox("Select Tone:", ["Neutral", "Suspenseful", "Inspiring"])
st.sidebar.info("All tone rewrites powered by IBM Granite LLM.")

input_mode = st.radio("Input Mode:", ["Paste Text", "Upload .txt File"])

if input_mode == "Paste Text":
    user_text = st.text_area("Enter your text here:", height=250)
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    user_text = uploaded_file.read().decode("utf-8") if uploaded_file else ""

if st.button("🎨 Generate Tone & Audio"):
    if not user_text.strip():
        st.warning("Please enter or upload text first.")
    else:
        with st.spinner("Rewriting text using IBM Granite..."):
            rewritten_text = rewrite_text_with_tone(user_text, tone)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📜 Original Text")
            st.text_area("Original", user_text, height=400)
        with col2:
            st.subheader(f"✨ {tone} Version (Granite LLM)")
            st.text_area("Rewritten", rewritten_text, height=400)

        with st.spinner("🎤 Generating voice narration..."):
            audio_path = generate_audio(rewritten_text)

        st.audio(audio_path, format="audio/mp3")
        with open(audio_path, "rb") as f:
            st.download_button("⬇️ Download MP3", f, file_name="EchoVerse_Narration.mp3")

        os.remove(audio_path)


