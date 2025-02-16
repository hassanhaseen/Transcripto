import streamlit as st
import whisper
import tempfile
import os
import warnings
import torch
import soundfile as sf
import numpy as np
import noisereduce as nr
from textblob import TextBlob
from transformers import pipeline
from gtts import gTTS
from deep_translator import GoogleTranslator
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder  # ✅ Replaces pyaudio for recording

# Suppress Whisper FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Explicitly set FFmpeg path
os.environ["PATH"] += os.pathsep + "/usr/bin"

# Set page config
st.set_page_config(page_title="Transcripto", page_icon="✍️", layout="centered")

# ✅ Load Whisper Model (CPU Mode for Compatibility)
@st.cache_resource
def load_model():
    return whisper.load_model("small", device="cpu")  # Running on CPU to avoid memory issues

model = load_model()

# ✅ Transcription Function
def transcribe_audio(file_path):
    if os.stat(file_path).st_size == 0:
        return "⚠️ Error: Empty audio file. Please try again."

    try:
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"⚠️ Error during transcription: {str(e)}"

# ✅ UI: Select Mode
st.sidebar.title("🎙️ Transcripto - AI-Powered Speech-to-Text")
mode = st.sidebar.radio("Choose Mode", ["🎤 Record & Transcribe", "📂 Upload & Transcribe"])

if mode == "📂 Upload & Transcribe":
    uploaded_file = st.file_uploader("📥 Upload an Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

    if uploaded_file:
        st.audio(uploaded_file, format="audio/mp3")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_audio_path = temp_file.name

        if st.button("🎬 Start Transcription"):
            with st.spinner("⏳ Transcribing..."):
                transcribed_text = transcribe_audio(temp_audio_path)
                st.success("✅ Transcription Complete!")
                st.text_area("📜 Transcribed Text:", transcribed_text, height=150)
                os.remove(temp_audio_path)

elif mode == "🎤 Record & Transcribe":
    st.write("🎙 Click the button below to start recording.")

    # ✅ `mic_recorder()` now correctly returns an audio file
    audio_data = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording", key="mic")

    if audio_data and isinstance(audio_data, dict) and "bytes" in audio_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data["bytes"])
            temp_audio_path = temp_file.name

        st.audio(temp_audio_path, format="audio/wav")

        with st.spinner("⏳ Transcribing..."):
            transcribed_text = transcribe_audio(temp_audio_path)
            st.success("✅ Transcription Complete!")
            st.text_area("📜 Transcribed Text:", transcribed_text, height=150)
            os.remove(temp_audio_path)
