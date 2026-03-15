import streamlit as st
from dotenv import load_dotenv
from crewai import Crew, Process

# 1. Load environment variables
load_dotenv()

# Import custom modules
from src.agents import create_researcher_agent, create_policy_analyst_agent
from src.tasks import create_research_task, create_analysis_task

# 2. Configure the Streamlit page
st.set_page_config(page_title="Canada AI Strategy Consultant", page_icon="🍁", layout="centered")
st.title("🍁 Canada AI Strategy Consultant")
st.markdown("Ask me anything about Canada's AI innovation ecosystem, competitiveness, and recommended policy steps.")

# 3. Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle new user input
if prompt := st.chat_input("E.g., How can Canada improve its AI talent retention?"):
    
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 5. Orchestrate the CrewAI workflow
    with st.chat_message("assistant"):
        with st.spinner("Analyzing AI strategy data..."):
            
            # Initialize the agents
            researcher = create_researcher_agent()
            analyst = create_policy_analyst_agent()
            
            # Initialize the tasks with the user's prompt
            research_task = create_research_task(researcher, prompt)
            analysis_task = create_analysis_task(analyst, prompt)
            
            # Assemble the Crew
            strategy_crew = Crew(
                agents=[researcher, analyst],
                tasks=[research_task, analysis_task],
                process=Process.sequential, # Tasks run one after the other
                verbose=True
            )
            
            # Execute the workflow
            try:
                result = strategy_crew.kickoff()
                final_response = getattr(result, 'raw', str(result))
            except Exception as e:
                final_response = f"An error occurred during analysis: {e}"
            
            # Display the result
            st.markdown(final_response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})