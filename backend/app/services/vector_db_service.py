import chromadb

# Create a ChromaDB client
client = chromadb.PersistentClient(path="vector_db")

# Create (or load) a collection
collection = client.get_or_create_collection(
    name="enterprise_documents"
)


def store_embeddings(chunks, embeddings):

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        collection.add(
            ids=[str(index)],
            documents=[chunk],
            embeddings=[embedding]
        )

    return "Embeddings stored successfully."