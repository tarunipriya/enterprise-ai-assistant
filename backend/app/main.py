from fastapi import FastAPI

from app.routes.chat import router as chat_router
from app.routes.upload import router as upload_router

app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description="LLM-powered AI Assistant with RAG capabilities",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to Enterprise AI Knowledge Assistant"
    }


# Register API Routes
app.include_router(chat_router)
app.include_router(upload_router)