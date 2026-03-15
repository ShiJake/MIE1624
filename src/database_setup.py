import os
import glob
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import LangChain loaders for unstructured data types
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader
)
import pandas as pd

# 1. Load environment variables
load_dotenv()

# 2. Dynamically determine paths
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
                loader = UnstructuredWordDocumentLoader(file_path)
                documents.extend(loader.load())
                
            elif ext == ".pptx":
                loader = UnstructuredPowerPointLoader(file_path)
                documents.extend(loader.load())
                
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            
            # Handle structured tabular documents
            elif ext == ".csv":
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    content = " | ".join([f"{col}: {val}" for col, val in row.items()])
                    documents.append(Document(page_content=content, metadata={"source": file_name}))
                    
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
    # =====================================================================
    # FUTURE IMPLEMENTATION: Uncomment the line below to use real files
    # documents = load_diverse_files(RAW_DATA_PATH)
    # =====================================================================
    
    # 3. Initialize Dummy Data (CURRENT FALLBACK FOR TESTING)
    # If the real documents list is empty (or the line above is commented out), use dummy data
    # documents = [ ... dummy data ... ]
    # (Leaving this implementation active so your code runs today)
    
    print("Initializing dummy data for testing...")
    dummy_texts = [
        "Part 1 Analysis: A 2025 KPMG study shows Canada ranks 44th in AI literacy out of 47 countries.",
        "Part 2 Strategy: The enhanced strategy expands the $2 billion Canadian Sovereign AI Compute Strategy to prioritize domestic startups.",
        "Part 3 Practical Steps: We recommend modifying the Mitacs research internship program to increase funding for AI-specific industry-academic collaborations.",
        "Part 4 Storytelling: The PR campaign 'Canada - the country of AI innovations' focuses on ethical AI and attracting international investments."
    ]
    documents = [Document(page_content=text) for text in dummy_texts]

    # 4. Configure the Text Splitter
    print(f"Chunking {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Increased chunk size to preserve context in longer reports
        chunk_overlap=150,
        length_function=len
    )
    chunked_docs = text_splitter.split_documents(documents)
    
    # 5. Generate Embeddings and Build the FAISS Index
    print("Generating embeddings and building the FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_db = FAISS.from_documents(chunked_docs, embeddings)
    
    # 6. Save the database
    os.makedirs(DB_PATH, exist_ok=True)
    vector_db.save_local(DB_PATH)
    print(f"\nSuccess! Vector database saved to:\n{DB_PATH}")

if __name__ == "__main__":
    build_and_save_database()