import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Define the relative path to your saved vector database
# This points to the data/vector_store/ directory in your Git repo
DB_PATH = os.path.join(os.path.dirname(__dirname__), "data", "vector_store")

def load_knowledge_base():
    """
    Loads the local FAISS vector database containing insights from Parts 1-4.
    Returns the loaded database object.
    """
    # 1. Initialize the embedding model used to create the original database
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 2. Check if the database directory exists to prevent crash errors
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Vector database not found at {DB_PATH}. "
            "Please ensure you have generated the database from your Part 1-4 data."
        )
    
    # 3. Load the database from the local directory
    # allow_dangerous_deserialization is required to load local pickle (.pkl) files safely
    vector_db = FAISS.load_local(
        folder_path=DB_PATH, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    
    return vector_db