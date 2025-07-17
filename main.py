from fastapi import FastAPI

from app import agent_api
from app.embeddings import upload_file
from contextlib import asynccontextmanager

from app.graph_builder import get_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = get_graph()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(upload_file.router)
app.include_router(agent_api.router)