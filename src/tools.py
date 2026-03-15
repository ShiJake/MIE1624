from crewai.tools import tool
from src.database import load_knowledge_base

# 1. Initialize the database connection globally within this module
try:
    vector_db = load_knowledge_base()
except FileNotFoundError as e:
    print(f"Warning: {e}")
    vector_db = None

# 2. Define the tool using the CrewAI decorator
@tool("Search AI Strategy Database")
def search_strategy_database(query: str) -> str:
    """
    Useful for searching the local vector database containing analysis, reports, 
    and practical steps regarding Canada's AI strategy from Parts 1-4.
    Input should be a specific search query string.
    """
    if vector_db is None:
        return "Error: The vector database is not loaded. Please ensure data is processed."
    
    # Perform a similarity search in the FAISS index
    # k=3 retrieves the top 3 most relevant chunks of text to provide sufficient context
    results = vector_db.similarity_search(query, k=3)
    
    if not results:
        return "No relevant information found in the database."
    
    # Combine the retrieved text chunks into a single readable string for the AI agent
    context = "\n\n--- Next Retrieved Segment ---\n\n".join([doc.page_content for doc in results])
    
    return context