from app.services.vector_db_service import collection
from app.services.embedding_service import model


def retrieve_chunks(query: str, n_results: int = 3):

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results["documents"][0]