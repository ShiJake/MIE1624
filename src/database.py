import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Define the relative path to your saved vector database
# This points to the data/vector_store/ directory in your Git repo
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "vector_store")

def load_knowledge_base():
    """
    Loads the local FAISS vector database containing insights from Parts 1-4.
    Returns the loaded database object.
    """
    # 1. Initialize the embedding model used to create the original database
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
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