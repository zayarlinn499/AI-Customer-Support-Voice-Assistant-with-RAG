# AI Customer Support Voice Assistant with RAG

An end-to-end AI-powered voice assistant that automates customer support using Speech-to-Text (STT), Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), and Text-to-Speech (TTS).

Instead of relying solely on pretrained language model knowledge, the assistant retrieves relevant information from a company's private knowledge base before generating a response. This significantly reduces hallucinations and ensures answers are grounded in official company documentation.

---

# Features

* Real-time voice interaction through a microphone
* Speech-to-Text using Faster Whisper
* Retrieval-Augmented Generation (RAG)
* Semantic search with Qdrant Vector Database
* AI reasoning using Groq Llama 3.3 70B
* Natural voice responses using ElevenLabs
* PDF document ingestion
* Context-aware customer support responses
* Hallucination reduction using retrieved company knowledge
* Streamlit web interface

---

# Project Overview

The application simulates an AI customer support representative capable of answering frequently asked questions using company documents.

Users speak naturally into the microphone.

The system:

1. Records the user's voice.
2. Converts speech into text.
3. Searches a vector database containing company documents.
4. Retrieves the most relevant document chunks.
5. Uses a Large Language Model to generate an answer based only on retrieved information.
6. Converts the answer into natural speech.
7. Plays the response back to the user.

---

# System Architecture

```text
                Customer Voice
                      │
                      ▼
            Speech Recording (Microphone)
                      │
                      ▼
            Faster Whisper (Speech-to-Text)
                      │
                      ▼
               User Question (Text)
                      │
                      ▼
        Sentence Transformer Embeddings
                      │
                      ▼
         Qdrant Vector Database Search
                      │
                      ▼
        Retrieved Company Document Chunks
                      │
                      ▼
        Groq Llama 3.3 70B (LLM)
                      │
                      ▼
             AI Generated Response
                      │
                      ▼
          ElevenLabs Text-to-Speech
                      │
                      ▼
             Natural Voice Response
```

---

# Technology Stack

| Category              | Technology                             |
| --------------------- | -------------------------------------- |
| Programming Language  | Python                                 |
| AI Framework          | LangChain                              |
| Large Language Model  | Groq Llama 3.3 70B Versatile           |
| Speech-to-Text        | Faster Whisper                         |
| Text-to-Speech        | ElevenLabs                             |
| Embeddings            | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database       | Qdrant                                 |
| User Interface        | Streamlit                              |
| Environment Variables | python-dotenv                          |

---

# Project Structure

```text
AI-Customer-Support-Voice-Assistant-with-RAG/

│
├── audio/
│   ├── input.wav
│   └── response.mp3
│
├── data/
│   ├── Company Information.pdf
│   ├── Refund Policy.pdf
│   └── Shipping Policy.pdf
│
├── qdrant_db/
│
├── src/
│   ├── rag/
│   │   ├── ingest.py
│   │   ├── search.py
│   │   └── chat.py
│   │
│   └── voice/
│       ├── recorder.py
│       ├── stt.py
│       ├── tts.py
│       └── voice_agent.py
│
├── app.py
├── requirements.txt
├── .env.example
└── README.md
```

---

# System Workflow

```text
User speaks
      │
      ▼
Microphone Recording
      │
      ▼
Speech-to-Text (Whisper)
      │
      ▼
Question Text
      │
      ▼
Vector Search (Qdrant)
      │
      ▼
Relevant Company Documents
      │
      ▼
LLM Response Generation (Groq)
      │
      ▼
Text-to-Speech (ElevenLabs)
      │
      ▼
Voice Response
```

---

# Knowledge Base

The assistant answers questions using company documents stored in PDF format.

Current sample knowledge base:

* Company Information
* Refund Policy
* Shipping Policy

Each PDF is:

1. Loaded
2. Split into chunks
3. Converted into embeddings
4. Stored in Qdrant Vector Database

---

# Example Questions

* What is the refund period?
* Where is the company headquarters?
* How long does shipping take?
* When are refunds processed?
* Which items are not eligible for refunds?

---

# Prompt Engineering

The assistant is instructed to:

* Answer only from retrieved company documents.
* Avoid generating unsupported information.
* Inform the user when information is unavailable.
* Generate concise and professional customer support responses.

---

# Current Capabilities

* Voice Input
* Speech Recognition
* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Company Knowledge Base
* AI Response Generation
* Voice Output
* Streamlit User Interface

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Customer-Support-Voice-Assistant-with-RAG.git
```

Navigate into the project:

```bash
cd AI-Customer-Support-Voice-Assistant-with-RAG
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

---

# Running the Project

Generate the vector database (first run):

```bash
python src/rag/ingest.py
```

Run the Streamlit interface:

```bash
streamlit run app.py
```

Or run the terminal version:

```bash
python src/voice/voice_agent.py
```

---

# Testing

The assistant has been tested using:

* Refund Policy
* Company Information
* Shipping Policy

Example:

**Question**

> What is the refund period?

**Answer**

> Customers may request a refund within 30 calendar days of receiving their order.

---

# Future Improvements

* Continuous conversation mode
* Conversation memory using LangGraph
* Live streaming speech recognition
* Twilio phone call integration
* Docker deployment
* Authentication and user management
* Cloud deployment (Azure, AWS, GCP)
* Multi-language support
* Admin dashboard
* Analytics and conversation logging

---

# Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Semantic Search
* Speech Recognition
* Speech Synthesis
* Large Language Models
* LangChain
* AI Application Development
* Prompt Engineering
* End-to-End AI System Integration

---

# License

This project is licensed under the MIT License.

---

# Author

**Zayar Linn**

AI & Full-Stack Developer

If you found this project helpful, consider starring the repository.
