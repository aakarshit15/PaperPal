from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store

def save_vector_store(vector_store, path="./faiss_index"):
    vector_store.save_local(path)

def load_vector_store(path="./faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(path):
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None