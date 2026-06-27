import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from elevenlabs.client import ElevenLabs
import pypdf
import re


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
INPUT_AUDIO = AUDIO_DIR / "ui_input.wav"
OUTPUT_AUDIO = AUDIO_DIR / "ui_response.mp3"
QDRANT_PATH = BASE_DIR / "qdrant_db"

AUDIO_DIR.mkdir(exist_ok=True)

@st.cache_data
def get_document_texts():
    docs_dir = BASE_DIR / "data"
    docs = {}
    if docs_dir.exists():
        for pdf_file in docs_dir.glob("*.pdf"):
            try:
                reader = pypdf.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + " "
                
                # Format into standard paragraphs
                text = re.sub(r'\s+', ' ', text).strip()
                text = text.replace(' ● ', '\n● ')
                text = text.replace(' • ', '\n• ')
                text = re.sub(r' (\d+\.) ', r'\n\1 ', text)
                text = re.sub(r'(?<=[a-z][.!?])\s+(?=[A-Z])', '\n\n', text)
                
                docs[pdf_file.stem] = text
            except Exception as e:
                pass
    return docs


@st.cache_resource
def load_whisper():
    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )


@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=str(QDRANT_PATH),
        collection_name="company_docs"
    )


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile"
    )


def speech_to_text(audio_path):
    whisper = load_whisper()

    segments, info = whisper.transcribe(str(audio_path))

    question = ""

    for segment in segments:
        question += segment.text

    return question.strip()


def get_rag_answer(question):
    vector_store = load_vector_store()
    llm = load_llm()

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a professional customer support assistant.

Answer ONLY from the provided context.

If the answer is not found in the context, say:
"I couldn't find that information in the company knowledge base."

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content


def text_to_speech(text):
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is missing in .env")

    client = ElevenLabs(api_key=api_key)

    audio = client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text=text,
        model_id="eleven_multilingual_v2"
    )

    with open(OUTPUT_AUDIO, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return OUTPUT_AUDIO


st.set_page_config(
    page_title="AI Voice Customer Support",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS for a clean, professional UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply font to primary text elements */
    html, body, .stApp, p, h1, h2, h3, h4, h5, h6, div, span, li, button {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Ensure Streamlit icons keep their font */
    .stIcon, .material-symbols-rounded, [class*="Icon"] {
        font-family: 'Material Symbols Rounded' !important;
    }
    
    /* Main Background Gradient */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #1e1b4b, #0f172a 30%, #020617 70%, #000000);
        color: #f8fafc;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Header Typography */
    .main-header {
        background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        margin-top: -30px;
        margin-bottom: 0px;
        padding-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 300;
        margin-bottom: 40px;
    }
    
    /* Chat Messages Glassmorphism */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, border 0.3s ease;
    }
    
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Audio Inputs and Player */
    [data-testid="stAudioInput"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.75rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 20px -10px rgba(129, 140, 248, 0.5) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 15px 25px -10px rgba(129, 140, 248, 0.7) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "processing_key" not in st.session_state:
    st.session_state.processing_key = None
if "history" not in st.session_state:
    st.session_state.history = []

# Header
st.markdown("<h1 class='main-header'>AI Voice Support</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Ask a question about company information, refund policy, or shipping policy.</p>", unsafe_allow_html=True)

# Add Expander for Documents
docs = get_document_texts()
if docs:
    st.markdown("""
    <div style='background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
        <strong style='color: #38bdf8;'>💡 Helpful Tip:</strong> The AI uses example documents for company information to answer your questions about the company. You can ask questions from this documents that AI used by expanding the section below.
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📚 View Source Documents (Click to Expand)", expanded=False):
        tabs = st.tabs(list(docs.keys()))
        for tab, (title, content) in zip(tabs, docs.items()):
            with tab:
                st.markdown(f"<div style='background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; font-size: 0.95rem; line-height: 1.6; color: #cbd5e1; border: 1px solid rgba(255,255,255,0.05); margin-top: 10px;'>{content}</div>", unsafe_allow_html=True)

st.markdown("---")

# Chat Container
chat_container = st.container()

with chat_container:
    for msg in st.session_state.history:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(msg["question"])
        with st.chat_message("assistant", avatar="🎧"):
            st.write(msg["answer"])
            if msg.get("audio"):
                st.audio(msg["audio"], format="audio/mp3")

# Audio Input at the bottom
audio_value = st.audio_input(
    "Record your question here:",
    key=f"audio_input_{st.session_state.input_key}"
)

if audio_value is not None:
    if st.session_state.processing_key != st.session_state.input_key:
        # Prevent double-processing
        st.session_state.processing_key = st.session_state.input_key
        
        with open(INPUT_AUDIO, "wb") as f:
            f.write(audio_value.getbuffer())
            
        with chat_container:
            # User Message Processing
            with st.chat_message("user", avatar="🧑‍💻"):
                with st.spinner("Transcribing..."):
                    question = speech_to_text(INPUT_AUDIO)
                
                if question:
                    st.write(question)
                else:
                    st.warning("No speech detected. Please try again.")
            
            # Assistant Message Processing
            if question:
                with st.chat_message("assistant", avatar="🎧"):
                    with st.spinner("Searching knowledge base..."):
                        answer = get_rag_answer(question)
                        
                    with st.spinner("Generating voice response..."):
                        response_audio = text_to_speech(answer)
                        history_audio_path = AUDIO_DIR / f"response_{st.session_state.input_key}.mp3"
                        shutil.copy(response_audio, history_audio_path)
                        
                    st.write(answer)
                    st.audio(str(history_audio_path), format="audio/mp3", autoplay=True)
                    
                # Append to history
                st.session_state.history.append({
                    "question": question,
                    "answer": answer,
                    "audio": str(history_audio_path)
                })

    # Action Button to Ask Another Question
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎤 Ask Another Question", use_container_width=True, type="primary"):
            st.session_state.input_key += 1
            st.rerun()