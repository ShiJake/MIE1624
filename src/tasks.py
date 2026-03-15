from crewai import Task

def create_research_task(researcher_agent, query: str) -> Task:
    """
    Creates the initial task for the researcher to find data in the vector database.
    """
    return Task(
        description=(
            f"A user has asked the following question about Canada's AI strategy: '{query}'. "
            "Use your 'Search AI Strategy Database' tool to find the most relevant facts, "
            "data points, and context from the provided analytical reports and proposals. "
            "Extract the key information meticulously."
        ),
        expected_output="A detailed summary of raw facts and retrieved context relevant to the user's query.",
        agent=researcher_agent
    )

def create_analysis_task(analyst_agent, query: str) -> Task:
    """
    Creates the final task for the analyst to format the researcher's findings into a business response.
    """
    return Task(
        description=(
            f"Review the raw facts provided by the research team regarding the user's query: '{query}'. "
            "Synthesize this information into a clear, strategic, and concise answer. "
            "Do not invent new data; rely solely on the researcher's findings. "
            "Format the response professionally, as if speaking directly to a government official or business leader."
        ),
        expected_output="A polished, conversational, and strategic 1-2 paragraph response answering the user's query.",
        agent=analyst_agent
    )