from fastapi import APIRouter
from app.models.chat import ChatRequest
from app.services.retrieval_service import retrieve_chunks
from app.services.llm_service import generate_response

router = APIRouter()


@router.post("/chat")
def chat(request: ChatRequest):

    # Retrieve relevant chunks from ChromaDB
    retrieved_chunks = retrieve_chunks(request.message)

    # If no relevant chunks are found
    if not retrieved_chunks:
        return {
            "response": "No relevant information found in the uploaded documents."
        }

    # Convert retrieved chunks into a single context string
    context = "\n\n".join(retrieved_chunks)

    # Generate answer using the retrieved context
    answer = generate_response(
        question=request.message,
        context=context
    )

    return {
        "question": request.message,
        "retrieved_chunks": retrieved_chunks,
        "response": answer
    }