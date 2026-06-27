import os
from pathlib import Path

from dotenv import load_dotenv
from faster_whisper import WhisperModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from elevenlabs.client import ElevenLabs
from recorder import record_audio


load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_AUDIO = BASE_DIR / "audio" / "input.wav"
OUTPUT_AUDIO = BASE_DIR / "audio" / "response.mp3"
QDRANT_PATH = BASE_DIR / "qdrant_db"
COLLECTION_NAME = "company_docs"


def speech_to_text(audio_path):
    print("Loading Whisper...")

    whisper = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    segments, info = whisper.transcribe(str(audio_path))

    question = ""

    for segment in segments:
        question += segment.text

    return question.strip()


def get_rag_answer(question):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=str(QDRANT_PATH),
        collection_name=COLLECTION_NAME
    )

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

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


def text_to_speech(text, output_path):
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is missing in .env")

    client = ElevenLabs(
        api_key=api_key
    )

    audio = client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        text=text,
        model_id="eleven_multilingual_v2"
    )

    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    print("\nAudio saved:")
    print(output_path)


def play_audio(audio_path):
    audio_path = Path(audio_path)

    if audio_path.exists():
        os.startfile(str(audio_path))
    else:
        print("Audio file not found:", audio_path)


def main():
    record_audio(str(INPUT_AUDIO))

    question = speech_to_text(INPUT_AUDIO)

    print("\nQuestion:")
    print(question)

    if not question:
        print("No speech detected. Please try again.")
        return

    answer = get_rag_answer(question)

    print("\nAnswer:")
    print(answer)

    text_to_speech(answer, OUTPUT_AUDIO)

    play_audio(OUTPUT_AUDIO)


if __name__ == "__main__":
    main()