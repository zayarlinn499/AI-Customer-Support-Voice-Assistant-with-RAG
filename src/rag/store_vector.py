from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load PDFs
all_docs = []

for pdf_file in Path("data").glob("*.pdf"):
    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()

    all_docs.extend(docs)

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(all_docs)

print("Chunks:", len(chunks))

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Qdrant database
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    path="./qdrant_db",
    collection_name="company_docs"
)

print("Stored successfully!")