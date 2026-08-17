import operator
from typing import TypedDict, List, Annotated, Dict, Any

class ResearchState(TypedDict):
    original_query: str
    sub_questions: List[str]
    research_data: Annotated[List[Dict[str, Any]], operator.add]
    research_score: float
    evaluation_feedback: str
    research_iterations: int
    draft_report: str
    final_report: str
