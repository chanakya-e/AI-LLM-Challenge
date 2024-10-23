from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class SlackNotifier:
    """Handles sending notifications to Slack."""
    
    def __init__(self, token: str):
        self.client = WebClient(token=token)
    
    def post_message(self, channel: str, message: str) -> None:
        """Posts a message to a Slack channel."""
        try:
            response = self.client.chat_postMessage(channel=channel, text=message)
        except SlackApiError as e:
            raise Exception(f"Error posting to Slack: {e.response['error']}")
