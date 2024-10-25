import streamlit as st
from ai_agent import AIAgent  # Import the AI agent class
from config.config import Config  # Config file for defaults

# Streamlit Interface
st.title("AI PDF Question Answering Agent")

# File Uploader
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

# Question Input
questions = st.text_area("Enter questions (one per line)").split('\n')  # Split by line breaks

# Mandatory Slack API Token in Sidebar
st.sidebar.header("Configuration")
slack_token = st.sidebar.text_input("Enter Slack API Token (mandatory)", value=Config.SLACK_API_TOKEN, type="password")
slack_channel = st.sidebar.text_input("Enter Slack Channel", value=Config.SLACK_CHANNEL)

# Optional Model Name
model_name = st.sidebar.text_input("Enter HuggingFace model name", value=Config.HUGGINGFACE_MODEL)

# Status display function
status_placeholder = st.empty()

def update_status(message):
    """Function to update the status message in Streamlit."""
    status_placeholder.text(message)

# Submit button
if st.button("Submit"):
    # Check if Slack API Token has been changed from the dummy value
    if slack_token == Config.SLACK_API_TOKEN:
        st.sidebar.error("Slack API Token is mandatory and cannot be the default dummy value.")
    elif uploaded_files and questions:
        # Clean up empty questions in case of accidental empty lines
        questions = [q.strip() for q in questions if q.strip()]  # Remove empty lines or spaces

        pdf_files = [uploaded_file.getvalue() for uploaded_file in uploaded_files]

        # Create AI Agent with user-provided or default configuration
        agent = AIAgent(pdf_files=pdf_files, model_name=model_name, slack_token=slack_token, slack_channel=slack_channel)
        
        # Process the PDF and questions while showing which question is being processed
        agent.process_and_notify(questions, update_status)
        st.success("Questions submitted and answers posted to Slack.")
        status_placeholder.empty()  # Clear the status message
    else:
        st.error("Please upload PDFs and enter questions.")
