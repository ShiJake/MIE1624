import os
import glob
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import LangChain loaders for unstructured data types
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)
import pandas as pd

# Load environment variables
load_dotenv()

# Dynamically determine paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "vector_store")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")

def load_diverse_files(directory_path: str) -> list[Document]:
    """
    Scans the target directory and applies the correct LangChain loader or 
    Pandas parsing logic based on the file extension.
    """
    documents = []
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found. Returning empty list.")
        return documents

    # Grab every file in the directory
    all_files = glob.glob(os.path.join(directory_path, "*.*"))
    
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        print(f"Processing: {file_name}...")
        
        try:
            # Handle unstructured text documents
            if ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
                documents.extend(loader.load())
            
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
                
            elif ext == ".pptx":
                loader = UnstructuredPowerPointLoader(file_path)
                documents.extend(loader.load())
                
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            
            # Handle structured tabular documents
            elif ext == ".csv":
                print("here")
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    documents.append(Document(page_content=content, metadata={"source": file_name}))
                print("here")
                    
            elif ext == ".xlsx":
                df = pd.read_excel(file_path)
                for _, row in df.iterrows():
                    content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    documents.append(Document(page_content=content, metadata={"source": file_name}))
            
            else:
                print(f"  -> Warning: Skipped unsupported file type ({ext})")
                
        except Exception as e:
            print(f"  -> Error loading {file_name}: {e}")
   
    return documents

def build_and_save_database():
    """
    Creates a FAISS vector database from raw files (or dummy data) and saves it locally.
    """

    documents = load_diverse_files(RAW_DATA_PATH)

    if not documents:
        print("No new documents found in data/raw. Exiting.")
        return

    # Configure the Text Splitter
    print(f"Chunking {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Increased chunk size to preserve context in longer reports
        chunk_overlap=150,
        length_function=len
    )
    chunked_docs = text_splitter.split_documents(documents)

    print(f"Generating embeddings for {len(chunked_docs)} chunks...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    batch_size = 15

    if os.path.exists(DB_PATH):
        print(f"--- Update Mode: Loading existing index from {DB_PATH} ---")
        vector_db = FAISS.load_local(
            DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        start_index = 0
    else:
        print("--- No existing index found. Initializing new database ---")
        # Initialize with a tiny slice if starting from scratch
        vector_db = FAISS.from_documents(chunked_docs[:15], embeddings)
        start_index = 15

    # Loop through the remaining chunks
    for i in range(start_index, len(chunked_docs), batch_size):
        batch = chunked_docs[i : i + batch_size]
        
        # Retry Logic: If we hit a 429, wait and try again
        success = False
        retries = 0
        while not success and retries < 5:
            try:
                vector_db.add_documents(batch)
                success = True
                
                # Progress update every 150 chunks
                if i % 150 == 0:
                    print(f"  -> Progress: {i}/{len(chunked_docs)} chunks processed...")
                    time.sleep(10) # Take a longer breath
                else:
                    time.sleep(2) # Small pause between every small batch
                    
            except Exception as e:
                if "429" in str(e):
                    retries += 1
                    wait_time = 30 * retries
                    print(f"  !! Rate limit (429) hit at chunk {i}. Waiting {wait_time}s (Retry {retries}/5)...")
                    time.sleep(wait_time)
                else:
                    print(f"  !! Fatal error at chunk {i}: {e}")
                    raise e

    # Save the completed database
    os.makedirs(DB_PATH, exist_ok=True)
    vector_db.save_local(DB_PATH)
    print(f"\nSuccess! Database updated and saved to {DB_PATH}")


if __name__ == "__main__":
    build_and_save_database()