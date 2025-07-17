import os, shutil
from typing import List

from fastapi import HTTPException, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from app.config import doc_upload_dir, embed_model, vector_db_dir

os.makedirs(doc_upload_dir, exist_ok=True)


async def upload_files_and_save(files: List[UploadFile] = File(...)):
    saved_files = []
    all_chunks = []

    print("filename size", len(files))
    for file in files:
        print("filename", file.filename)
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")

        # Save to disk
        file_path = os.path.join(doc_upload_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(file.filename)

        # Load and extract text
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Split pages into chunks (optional)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)

        # Update metadata with filename
        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        all_chunks.extend(chunks)

    if not all_chunks:
        raise HTTPException(status_code=500, detail="No content extracted.")

    # Embed and save to vector store
    Chroma.from_documents(documents=all_chunks, embedding=embed_model, persist_directory=vector_db_dir)

    return {"status": "indexed", "files": saved_files}


