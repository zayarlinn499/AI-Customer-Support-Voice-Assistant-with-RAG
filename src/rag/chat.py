import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# LLM
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "500")),
    temperature=float(os.getenv("GROQ_TEMPERATURE", "0.6"))
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    path="./qdrant_db",
    collection_name="company_docs"
)

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a warm, helpful customer support representative speaking to a customer.
Answer the customer's question using ONLY the provided context, but in a natural, polite, and human conversational tone without markdown, bullet points, or robotic phrases like "According to the context".

Context:
{context}

Customer Question:
{question}

Response:
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)