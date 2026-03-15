# MIE 1624 - Part 5: Canada AI Strategy Consultant Chatbot

## Project Overview
This repository contains the standalone LLM chatbot implementation . The application allows users to have a meaningful conversation about Canada's AI strategy, innovation ecosystem, and global competitiveness. 

To satisfy the advanced implementation requirements, this chatbot utilizes **CrewAI** to demonstrate multi-agent collaboration (Researcher and Policy Analyst agents), agentic workflows, and tool use via a LangChain/FAISS Retrieval-Augmented Generation (RAG) pipeline.

---

## Prerequisites
* **Python**: Version `3.13` or higher is recommended.
* **API Key**: An active Google API key (or equivalent for Claude/GPT if configured in `agents.py`).

---

## Step-by-Step Installation

### 1. Install Dependencies
Install the required packages, including Streamlit, CrewAI, and LangChain:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a file named `.env` in the root directory of the project. Open it and add your API key:
```bash
GOOGLE_API_KEY="your-api-key-here"
```


## Running the Application

### 1. Generate the Vector Database
Before launching the chatbot, the multi-agent system needs data to search. Run the data setup script to generate the FAISS vector index locally:
```bash
python src/database_setup.py
```

### Step 2: Launch the Streamlit Interface
Once the database is built, launch the front-end application:
```bash
python -m streamlit run app.py
```

Step 3: Interact with the Chatbot
* The command above will automatically open a new tab in your default web browser
* Type a question into the text box (e.g., "How can Canada improve its AI talent retention?") and press Enter.
* You will see the agents sequentially process the query before outputting a strategic response.