from typing import List

from fastapi import APIRouter, UploadFile, File

from app.embeddings.save_vector import upload_files_and_save

router = APIRouter()


@router.post("/upload-files/")
async def upload_files(files: List[UploadFile] = File(...)):
    result = await upload_files_and_save(files)
    return result


