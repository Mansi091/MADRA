import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import ResearchState

load_dotenv()

class SubQuestions(BaseModel):
    questions: list[str] = Field(
        description="A list of 2 to 4 distinct, highly specific sub-questions that need to be researched to answer the original query."
    )

def planner_node(state: ResearchState) -> dict:
    print("PLANNER NODE RUNNING")
    query = state.get("original_query")
    feedback = state.get("evaluation_feedback", "")
    evidence = state.get("evidence", [])
    
    compiled_research = ""
    for ev in evidence:
        compiled_research += f"\nSub-question: {ev.get('question')}\n"
        compiled_research += f"Claims Found: {len(ev.get('claims', []))}\n"
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    structured_llm = llm.with_structured_output(SubQuestions)
    
    prompt = f"""
    You are an expert Research Planner.
    Based on the original query and the research gathered so far, generate a list of highly specific sub-questions that need to be answered next.
    If the gathered research is empty, generate initial sub-questions.
    If the gathered research is partial, generate follow-up questions to fill the gaps.
    Generate a maximum of 3 sub-questions.
    
    CRITICAL INSTRUCTION: You MUST use the provided tool/function to output your response. 
    Do NOT output plain text or markdown. Call the tool with the 'sub_questions' field.
    
    Original Query: {query}
    
    Gathered Research:
    {compiled_research}
    
    Previous Research:
    {compiled_research}
    
    Previous Evaluation Feedback:
    {feedback}
    Focus especially on the missing information identified by the evaluator.
    Do not repeat questions that have already been sufficiently researched.
    """
    
    result = structured_llm.invoke(prompt)
    
    print(f"Generated {len(result.questions)} sub-questions.")
    for q in result.questions:
        print(f" - {q}")
        
    return {"sub_questions": result.questions}
