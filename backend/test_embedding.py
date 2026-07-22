from app.services.embedding_service import generate_embeddings

chunks = [
    "Python is a programming language.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "Deep Learning uses neural networks."
]

embeddings = generate_embeddings(chunks)

print("Number of embeddings:", len(embeddings))
print("Dimensions of first embedding:", len(embeddings[0]))
print("First 10 values:")
print(embeddings[0][:10])