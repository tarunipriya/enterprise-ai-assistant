from app.services.embedding_service import generate_embeddings
from app.services.vector_db_service import store_embeddings

chunks = [
    "Python is a programming language.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "Deep Learning uses neural networks."
]

embeddings = generate_embeddings(chunks)

result = store_embeddings(chunks, embeddings)

print(result)