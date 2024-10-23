import json
from typing import List

class AnswerConsolidator:
    """Consolidates questions with their corresponding answers into a JSON structure."""
    
    def consolidate_responses(self, questions: List[str], responses: List[str]) -> str:
        data = {"questions": []}
        for question, answer in zip(questions, responses):
            data["questions"].append({
                "question": question,
                "answer": answer
            })
        return json.dumps(data, indent=4)
