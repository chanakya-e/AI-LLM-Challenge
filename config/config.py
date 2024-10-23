import os

class Config:
    """Configuration for the AI agent."""
    
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "deepset/roberta-base-squad2")
    SLACK_API_TOKEN = os.getenv("SLACK_API_TOKEN","dummy")
    SLACK_CHANNEL = os.getenv("SLACK_CHANNEL","C07T3RT54MQ")
