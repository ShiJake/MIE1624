import os
from crewai import Agent
from langchain_openai import ChatOpenAI
from src.tools import search_strategy_database

# 1. Initialize the LLM that will power the agents
# We use ChatOpenAI here, which connects to the OPENAI_API_KEY in your .env file
# You can change this to Gemini or Claude using their respective LangChain wrappers
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def create_researcher_agent() -> Agent:
    """
    Creates an agent dedicated to retrieving factual information from the Parts 1-4 analysis.
    """
    return Agent(
        role="AI Strategy Data Researcher",
        goal="Accurately retrieve specific findings, data points, and context regarding Canada's AI strategy from the local database.",
        backstory=(
            "You are a meticulous data researcher at the University of Toronto. "
            "Your job is to search through reports, social media sentiment analysis, "
            "and proposed practical steps to find the exact information needed to answer user queries. "
            "You never guess; you only use the data provided in the database."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[search_strategy_database],
        llm=llm
    )

def create_policy_analyst_agent() -> Agent:
    """
    Creates an agent dedicated to synthesizing the researcher's data into business-friendly insights.
    """
    return Agent(
        role="Senior AI Policy Analyst",
        goal="Synthesize retrieved data into clear, strategic, and business-oriented recommendations.",
        backstory=(
            "You are a Senior Consultant advising the Government of Canada. "
            "You take the raw data provided by your research team and turn it into "
            "compelling, easy-to-understand strategic narratives. "
            "Your audience is business-oriented and interested in actionable insights."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[], # The analyst does not search; they only analyze what the researcher finds
        llm=llm
    )