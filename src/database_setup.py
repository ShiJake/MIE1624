import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1. Load environment variables to securely access the OpenAI API key
load_dotenv()

# 2. Dynamically determine the absolute path to the data/vector_store/ directory
# This ensures the script works regardless of where the user runs it from their terminal
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "vector_store")

def build_and_save_database():
    """
    Creates a FAISS vector database from placeholder Parts 1-4 data 
    and saves it to the local repository structure.
    """
    print("Initializing dummy data representing Parts 1-4...")
    
    # 3. Initialize Dummy Data representing the output of your course project
    dummy_texts = [
        "Part 1 Analysis: A 2025 KPMG study shows Canada ranks 44th in AI literacy out of 47 countries.",
        "Part 2 Strategy: The enhanced strategy expands the $2 billion Canadian Sovereign AI Compute Strategy to prioritize domestic startups.",
        "Part 3 Practical Steps: We recommend modifying the Mitacs research internship program to increase funding for AI-specific industry-academic collaborations.",
        "Part 4 Storytelling: The PR campaign 'Canada - the country of AI innovations' focuses on ethical AI and attracting international investments."
    ]
    
    # Convert raw text strings into LangChain Document objects
    documents = [Document(page_content=text) for text in dummy_texts]
    
    # 4. Configure the Text Splitter
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        length_function=len
    )
    chunked_docs = text_splitter.split_documents(documents)
    
    # 5. Generate Embeddings and Build the FAISS Index
    print("Generating embeddings and building the FAISS index (this may take a moment)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.from_documents(chunked_docs, embeddings)
    
    # 6. Save the database to the designated local directory
    os.makedirs(DB_PATH, exist_ok=True)
    vector_db.save_local(DB_PATH)
    
    print(f"\nSuccess! Vector database saved successfully to:\n{DB_PATH}")
    print("The files 'index.faiss' and 'index.pkl' have been generated.")

if __name__ == "__main__":
    build_and_save_database()