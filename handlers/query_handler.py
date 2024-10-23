from transformers import pipeline
from typing import List

class QueryHandler:
    """Handles interaction with the Hugging Face transformer model for querying text."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-distilled-squad", confidence_threshold: float = 0.3):
        """Initialize the query handler with a Hugging Face model."""
        self.qa_pipeline = pipeline("question-answering", model=model_name)
        self.confidence_threshold = confidence_threshold

    def query_model(self, text_chunks: List[str], question: str) -> str:
        """Queries the Hugging Face transformer model with a given question and text chunks.
        Returns the responses that cross the confidence threshold, sorted by confidence score. 
        If none meet the threshold, return 'Data Not Available'.
        """
        valid_responses = []  # Store answers that meet the confidence threshold

        for chunk in text_chunks:
            # Use the QA model to query the text chunk with the question
            response = self.qa_pipeline({
                "context": chunk,
                "question": question
            })

            # If the response's confidence is above the threshold, store the answer and score
            if response['score'] >= self.confidence_threshold:
                valid_responses.append({
                    "answer": response['answer'],
                    "score": response['score']
                })

        # If no valid answers were found, return "Data Not Available"
        if not valid_responses:
            return ["Data Not Available"]

        # Sort the valid responses by confidence score in descending order
        sorted_responses = sorted(valid_responses, key=lambda x: x['score'], reverse=True)

        # Return the answers sorted by their confidence score
        return [resp['answer'] for resp in sorted_responses]

