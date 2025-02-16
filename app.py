import streamlit as st
import whisper
import tempfile
import os
import warnings
import torch
import soundfile as sf
import numpy as np
from textblob import TextBlob
from transformers import pipeline
from gtts import gTTS
from deep_translator import GoogleTranslator
from pydub import AudioSegment  # For audio processing

# Suppress Whisper FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Set page config
st.set_page_config(page_title="Transcripto", page_icon="✍️", layout="centered")

# ✅ Load Whisper Model (CPU Mode for Compatibility)
@st.cache_resource
def load_model():
    return whisper.load_model("small", device="cpu")  # Running on CPU to avoid memory errors

model = load_model()

# ✅ Preprocess Audio (Convert to Mono & 16kHz)
def preprocess_audio(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        processed_audio_path = audio_path.replace(".mp3", "_processed.wav")
        audio.export(processed_audio_path, format="wav")
        return processed_audio_path
    except Exception as e:
        st.error(f"⚠ Audio Processing Failed: {str(e)}")
        return audio_path

# ✅ Transcription Function
def transcribe_audio(file_path):
    if os.stat(file_path).st_size == 0:
        return "⚠️ Error: Empty audio file. Please try again."

    file_path = preprocess_audio(file_path)  # Ensure correct format

    try:
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        return f"⚠️ Error during transcription: {str(e)}"

# ✅ Sentiment Analysis Function
def analyze_sentiment(text):
    if not text.strip():
        return "Neutral"

    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "😊 Positive"
    elif polarity < 0:
        return "😡 Negative"
    else:
        return "😐 Neutral"

# ✅ Translation Function
def translate_text(text, target_language):
    try:
        return GoogleTranslator(source="auto", target=target_language).translate(text)
    except Exception as e:
        return f"⚠ Translation Error: {str(e)}"

# ✅ UI: File Upload
st.title("🎙️ Transcripto - AI-Powered Speech-to-Text")
uploaded_file = st.file_uploader("📥 Upload an Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

# ✅ UI: Select Language
language_options = {"English": "en", "Urdu": "ur", "Hindi": "hi", "French": "fr", "Spanish": "es"}
selected_language = st.selectbox("🌍 Select Transcription Language", list(language_options.keys()))
language_code = language_options[selected_language]

# ✅ UI: Translate to Another Language
translation_options = {"No Translation": None, "English": "en", "Urdu": "ur", "Spanish": "es", "French": "fr"}
selected_translation = st.selectbox("🌍 Translate To:", list(translation_options.keys()))

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_audio_path = temp_file.name

    if st.button("🎬 Start Transcription"):
        with st.spinner(f"Transcribing in {selected_language}... Please wait."):
            try:
                transcribed_text = transcribe_audio(temp_audio_path)
                sentiment_result = analyze_sentiment(transcribed_text)

                if translation_options[selected_translation]:
                    translated_text = translate_text(transcribed_text, translation_options[selected_translation])
                else:
                    translated_text = "No translation selected."

                # ✅ Display Results
                st.success("✅ Transcription Complete!")
                st.subheader("📜 Transcribed Text:")
                st.text_area("Text:", transcribed_text, height=150)

                st.subheader("💬 Sentiment Analysis:")
                st.write(sentiment_result)

                if translation_options[selected_translation]:
                    st.subheader("🌍 Translated Text:")
                    st.text_area("Translation:", translated_text, height=100)

                st.download_button("⬇️ Download Transcription", transcribed_text, "transcription.txt", "text/plain")
            except Exception as e:
                st.error(f"⚠ Transcription Failed: {str(e)}")

    os.remove(temp_audio_path)
