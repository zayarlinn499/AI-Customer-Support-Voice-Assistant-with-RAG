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

    company_prompt = (
        "Customer call with ABC Electronics customer support. "
        "Topics include consumer electronics, smart home devices, computer accessories, "
        "shipping policy, delivery timelines, tracking, returns, warranties, and refund policies."
    )

    segments, info = whisper.transcribe(
        str(audio_path),
        initial_prompt=company_prompt,
        beam_size=5
    )

    question = ""

    for segment in segments:
        question += segment.text

    return question.strip()


def get_rag_answer(question):
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "500")),
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.6"))
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
        k=4
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a warm, helpful customer support representative for ABC Electronics speaking directly to a customer on a phone call.

The customer's question is transcribed from real-time speech and may be informal, slightly complex, rambling, or contain speech-to-text imperfections/typos.
Always interpret the customer's intent through the lens of ABC Electronics products, shipping, warranty, returns, and refund policies.

Guidelines for your response:
1. Decode the core intent: Overlook background noise, filler words ("um", "like", "you know"), or garbled transcription words by mapping what they said to the relevant ABC Electronics topic.
2. Grounded in facts: Answer accurately using the company context below. Never invent new policies.
3. Natural phone demeanor: Speak as the company itself in a warm, polite first-person voice ("We provide...", "You can return..."). Never say "According to the context" or "Based on the documents".
4. NO markdown or symbols: Never use asterisks, bullet points, numbered lists, or bold tags, because this response will be read aloud through text-to-speech.
5. Keep it concise: 2 to 4 spoken sentences that directly resolve the customer's question.
6. If the audio is completely unintelligible noise, or if the question cannot be answered from the context, respond gracefully:
   "I'm sorry, I didn't quite catch that clearly. Could you please clarify what you need help with regarding your ABC Electronics product or order?"

Context:
{context}

Customer Question:
{question}

Spoken Response:
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