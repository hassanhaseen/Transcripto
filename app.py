import streamlit as st
import whisper
import numpy as np
import sounddevice as sd
import queue
import tempfile
import os
import scipy.io.wavfile as wav
import warnings
from textblob import TextBlob
from googletrans import Translator  # Translation for sentiment analysis

# Suppress Whisper FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Set page config with new name & icon
st.set_page_config(page_title="Transcripto", page_icon="✍️", layout="centered")

# Load Whisper Model (cached)
@st.cache_resource
def load_model():
    return whisper.load_model("small")  # Using "small" for better balance

model = load_model()

# Audio Recording Params
samplerate = 44100  # Sample rate in Hz
channels = 1  # Mono recording
q = queue.Queue()

# Language selection dictionary
language_options = {
    "Auto Detect": None,
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
}

# Sidebar for navigation
st.sidebar.title("✍️ Transcripto")
st.sidebar.markdown("### Select Mode:")
app_mode = st.sidebar.radio("", ["🎙️ Record & Transcribe", "📂 File Upload"])

# Language Selection
st.sidebar.markdown("### Select Language:")
selected_language = st.sidebar.selectbox("Choose a Language", list(language_options.keys()))
language_code = language_options[selected_language]  # Get correct language code

# Translator for multi-language sentiment analysis
translator = Translator()

# Function to Transcribe Audio
def transcribe_audio(file_path, language_code):
    if os.stat(file_path).st_size == 0:
        return "⚠️ Error: Empty audio file. Please try again."

    try:
        if language_code is None:
            result = model.transcribe(file_path)
        else:
            result = model.transcribe(file_path, language=language_code)
        return result["text"]
    except Exception as e:
        return f"⚠️ Error during transcription: {str(e)}"

# Function to Analyze Sentiment (Works for all languages)
def analyze_sentiment(text, language_code):
    try:
        # Translate if the text is not in English
        if language_code != "en":
            translated_text = translator.translate(text, src=language_code, dest="en").text
        else:
            translated_text = text

        sentiment = TextBlob(translated_text).sentiment.polarity
        if sentiment > 0:
            return "😊 Positive"
        elif sentiment < 0:
            return "😡 Negative"
        else:
            return "😐 Neutral"
    except Exception as e:
        return f"⚠️ Sentiment Analysis Error: {str(e)}"

# Record & Transcribe Mode
if app_mode == "🎙️ Record & Transcribe":
    st.title("🎙️ Record & Transcribe")
    st.markdown("Record your voice and get a transcription.")

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata.copy())

    duration = st.slider("🎚️ Set Recording Duration (seconds)", min_value=1, max_value=60, value=5)

    if st.button("🎤 Start Recording"):
        st.success("🟢 Listening... Speak now!")

        with sd.InputStream(callback=callback, channels=channels, samplerate=samplerate):
            sd.rec(int(samplerate * duration), samplerate=samplerate, channels=channels, dtype=np.int16)
            sd.wait()

        if not q.empty():
            audio_data = np.concatenate([q.get() for _ in range(q.qsize())], axis=0)

            if np.max(audio_data) == 0:
                st.warning("⚠️ No sound detected. Please speak louder.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    wav.write(temp_audio.name, samplerate, audio_data)
                    temp_path = temp_audio.name

                st.info("⏳ Transcribing...")
                text = transcribe_audio(temp_path, language_code)

                if "Error" in text:
                    st.error(text)
                else:
                    sentiment = analyze_sentiment(text, language_code)
                    st.success("✅ Transcription Complete!")
                    st.text_area("📜 Transcribed Text", text, height=200, label_visibility="hidden", key="record_transcription")
                    st.write(f"**Sentiment Analysis:** {sentiment}")
                    st.download_button("⬇️ Download Transcription", text, "transcription.txt", "text/plain")
                os.remove(temp_path)
        else:
            st.warning("⚠️ No audio detected. Please try recording again.")

# File Upload Mode
else:
    st.title("📂 Upload an Audio File for Transcription")
    uploaded_file = st.file_uploader("📥 Select an Audio File (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        st.audio(temp_path, format=f"audio/{file_ext}")

        if st.button("🎬 Start Transcription"):
            st.info("⏳ Transcribing...")
            text = transcribe_audio(temp_path, language_code)

            if "Error" in text:
                st.error(text)
            else:
                sentiment = analyze_sentiment(text, language_code)
                st.success("✅ Transcription Complete!")
                with st.expander("📜 Show Transcribed Text", expanded=True):
                    st.text_area("Transcription", text, height=200, label_visibility="hidden", key="file_transcription")
                st.write(f"**Sentiment Analysis:** {sentiment}")
                st.download_button("⬇️ Download Transcription", text, "transcription.txt", "text/plain")
            os.remove(temp_path)
