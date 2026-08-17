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
    
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(SubQuestions)
    
    prompt = f"""
    You are a Research Planner. Your job is to break down the following complex query into 2 to 4 distinct sub-questions.
    These sub-questions will be given to parallel researchers to search the web.
    Make the questions highly specific and actionable for a search engine.
    
    Original Query: {query}
    """
    
    result = structured_llm.invoke(prompt)
    
    print(f"Generated {len(result.questions)} sub-questions.")
    for q in result.questions:
        print(f" - {q}")
        
    return {"sub_questions": result.questions}
