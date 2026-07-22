from fastapi import APIRouter, UploadFile, File
import shutil

from app.services.pdf_service import extract_text
from app.services.chunk_service import split_text

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    # Save uploaded file
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text from PDF
    text = extract_text(file_path)

    # Split text into chunks
    chunks = split_text(text)

    # Return response
    return {
        "filename": file.filename,
        "total_chunks": len(chunks),
        "chunks": chunks
    }