import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import torch
import base64
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="EchoVerse", layout="wide")

# IBM Watson credentials (replace with your actual keys)
IBM_TTS_APIKEY = "YOUR_WATSON_TTS_API_KEY"
IBM_TTS_URL = "https://api.us-south.text-to-speech.watson.cloud.ibm.com"

# Load Granite Model
@st.cache_resource
def load_granite_model():
    model_path = "ibm-granite/granite-3.3-8b-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, device

model, tokenizer, device = load_granite_model()

# Initialize Watson TTS
@st.cache_resource
def init_tts():
    authenticator = IAMAuthenticator(IBM_TTS_APIKEY)
    tts = TextToSpeechV1(authenticator=authenticator)
    tts.set_service_url(IBM_TTS_URL)
    return tts

tts = init_tts()

# ------------------ FUNCTIONS ------------------
def rewrite_text_with_tone(original_text, tone):
    """Tone-adaptive text rewriting using Granite."""
    prompt = f"Rewrite the following text in a {tone} tone, keeping the original meaning intact:\n\n{original_text}"

    conv = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        conv,
        return_tensors="pt",
        thinking=True,
        return_dict=True,
        add_generation_prompt=True
    ).to(device)

    set_seed(42)
    output = model.generate(**input_ids, max_new_tokens=2048)
    rewritten = tokenizer.decode(output[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return rewritten.strip()


def text_to_speech(text, voice="en-US_AllisonV3Voice"):
    """Generate audio narration using IBM Watson TTS."""
    with open("output.mp3", "wb") as audio_file:
        audio_file.write(
            tts.synthesize(
                text,
                voice=voice,
                accept="audio/mp3"
            ).get_result().content
        )
    return "output.mp3"


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:file/mp3;base64,{bin_str}" download="{os.path.basename(bin_file)}">⬇️ Download {file_label}</a>'
    return href

# ------------------ STREAMLIT UI ------------------
st.title("🎧 EchoVerse – Generative Audiobook Creator")
st.markdown("Transform your text into expressive, natural-sounding audio with customizable tone and voice.")

st.sidebar.header("⚙️ Settings")
tone = st.sidebar.selectbox("Select Tone", ["Neutral", "Suspenseful", "Inspiring"])
voice = st.sidebar.selectbox("Select Voice", ["en-US_AllisonV3Voice", "en-US_MichaelV3Voice", "en-US_LisaV3Voice"])

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
input_text = ""

if uploaded_file:
    input_text = uploaded_file.read().decode("utf-8")
else:
    input_text = st.text_area("Or paste your text here:", height=200)

if st.button("Generate Audiobook 🎙️"):
    if not input_text.strip():
        st.warning("Please provide some text to process.")
    else:
        with st.spinner("Rewriting text using IBM Granite..."):
            rewritten = rewrite_text_with_tone(input_text, tone)

        st.success("✅ Tone adaptation complete!")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 Original Text")
            st.text_area("", input_text, height=300)
        with col2:
            st.subheader(f"✨ {tone} Version")
            st.text_area("", rewritten, height=300)

        with st.spinner("Converting to voice with IBM Watson TTS..."):
            audio_path = text_to_speech(rewritten, voice)

        st.audio(audio_path)
        st.markdown(get_binary_file_downloader_html(audio_path, 'Narration (MP3)'), unsafe_allow_html=True)
        st.success("🎉 Audio narration ready!")

st.markdown("---")
st.caption("Built with 💡 IBM Granite + Watson TTS + Streamlit | EchoVerse © 2025")
