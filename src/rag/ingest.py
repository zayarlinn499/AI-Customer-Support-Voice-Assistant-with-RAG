from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

text = "Customers may request refunds within 30 days."

embedding = model.encode(text)

print("Embedding Length:", len(embedding))

print("\nFirst 10 Numbers:")
print(embedding[:10])