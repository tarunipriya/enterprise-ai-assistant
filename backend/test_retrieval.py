from app.services.retrieval_service import retrieve_chunks

results = retrieve_chunks("What is Artificial Intelligence?")

print("Retrieved Chunks:\n")

for i, chunk in enumerate(results, start=1):
    print(f"{i}. {chunk}\n")