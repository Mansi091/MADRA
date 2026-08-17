from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import ResearchState

class ReflectorOutput(BaseModel):
    score: float = Field(description="A score from 0.0 to 10.0 grading the completeness of the research.")
    feedback: str = Field(description="Detailed feedback on what information is missing or needs further investigation.")

def reflector_node(state: ResearchState) -> dict:
    print("REFLECTOR NODE RUNNING")
    query = state.get("original_query")
    research_data = state.get("research_data", [])
    current_iterations = state.get("research_iterations", 0)
    
    compiled_research = ""
    for idx, data in enumerate(research_data):
        compiled_research += f"\nSub-question {idx+1}: {data['question']}\n"
        compiled_research += f"Summary: {data['summary']}\n"
    
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(ReflectorOutput)
    
    prompt = f"""
    You are an expert Research Evaluator. Review the gathered research against the original user query.
    Score the research out of 10 based on completeness, source quality, and coverage.
    If the research is missing key aspects of the original query, explain exactly what is missing in the feedback.
    
    Original Query: {query}
    
    Gathered Research:
    {compiled_research}
    """
    
    result = structured_llm.invoke(prompt)
    
    print(f"Reflector Score: {result.score}/10.0")
    print(f"Reflector Feedback: {result.feedback}")
    
    return {
        "research_score": result.score,
        "evaluation_feedback": result.feedback,
        "research_iterations": current_iterations + 1
    }
