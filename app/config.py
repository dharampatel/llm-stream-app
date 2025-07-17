import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
doc_upload_dir = os.getenv("UPLOAD_DIR")
vector_db_dir = os.getenv("VECTOR_DB_DIR")

llm = ChatGoogleGenerativeAI(api_key=google_api_key, model='gemini-2.0-flash')
embed_model = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model='models/embedding-001')

def get_retriever():
    vectorstore = Chroma(persist_directory="vector_store", embedding_function=embed_model)
    return vectorstore.as_retriever()
