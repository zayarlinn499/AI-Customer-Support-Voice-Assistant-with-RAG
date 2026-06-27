from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Same embedding model used during storage
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to existing Qdrant database
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    path="./qdrant_db",
    collection_name="company_docs"
)

query = "What is the refund period?"

results = vector_store.similarity_search(
    query=query,
    k=3
)

print("\nQUESTION:")
print(query)

print("\nRESULTS:")

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("-" * 50)
    print(doc.page_content)