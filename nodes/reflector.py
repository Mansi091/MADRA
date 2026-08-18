from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import ResearchState

class ReflectionResult(BaseModel):
    score: float = Field(description="A score from 0.0 to 10.0 grading the completeness and quality of the evidence.")
    feedback: str = Field(description="Detailed feedback on what information is missing or needs further investigation.")

def reflector_node(state: ResearchState) -> dict:
    print("\n[AGENT 3: VERIFIER & REFLECTOR] RUNNING")
    query = state.get("original_query")
    evidence_list = state.get("evidence", [])
    current_iterations = state.get("research_iterations", 0)
    
    compiled_evidence = ""
    for ev in evidence_list:
        compiled_evidence += f"\nSource [{ev['source_id']} - {ev['url']}]:\n"
        for claim in ev.get("claims", []):
            compiled_evidence += f"  - {claim}\n"
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    structured_llm = llm.with_structured_output(ReflectionResult)
    
    prompt = f"""
    You are an expert Research Evaluator.
    Review the gathered evidence against the original query.
    
    CRITICAL INSTRUCTION: You MUST use the provided tool/function to structure your response. 
    Do NOT output plain text or markdown. Call the tool with the 'score' and 'feedback' fields.
    
    Original Query: {query}
    
    Gathered Evidence & Claims:
    {compiled_evidence}
    """
    
    import time
    time.sleep(2)
    result = structured_llm.invoke(prompt)
    
    print(f"  -> Score: {result.score}/10.0")
    print(f"  -> Feedback: {result.feedback}")
    
    return {
        "research_score": result.score,
        "evaluation_feedback": result.feedback,
        "research_iterations": current_iterations + 1
    }
