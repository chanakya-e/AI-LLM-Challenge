# AI Agent for PDF Querying

## What This Project Does

This project is about an AI agent that helps you get answers from PDF files. You can ask it questions, and it will search through PDF documents to find the relevant answers. It sends the answers directly to your Slack workspace.

## How It Works

1. The AI agent reads your PDF files.
2. It leverages a large language model to understand the contents of the PDF.
3. You can query the agent with specific questions.
4. The agent searches through the PDF to find the relevant information and responds with the answer.
5. The AI agent integrates with Slack to send you the results of the query.

## What You Need

- **Python**: Ensure Python (version 3.8 or above) is installed on your system.
- **Slack Account**: You need a Slack workspace and API keys to connect the AI agent to Slack.
- **API Keys**: You will need API keys from Slack to enable communication.

## Installing Requirements

To install the necessary Python libraries, execute the following command:

```bash
pip install -r requirements.txt
```

## Executing the code

You can run the code using the following command

```bash
streamlit run app.py 
```

After executing the command, you will be able to click the links, allowing you to upload PDFs and ask questions directly through the browser.
