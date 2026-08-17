from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

def researcher_node(state: Dict[str, Any]) -> dict:
    print("RESEARCHER NODE RUNNING")
    
    sub_question = state.get("sub_question")
    original_query = state.get("original_query")
    
    print(f"Researching: {sub_question}")
    
    search_tool = DuckDuckGoSearchRun()
    search_results = search_tool.invoke(sub_question)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert researcher. Read the following raw search results and summarize the key findings that answer the specific research question. Ensure the summary is highly relevant to the overarching original query."),
        ("human", "Original Query: {original_query}\n\nSpecific Research Question: {sub_question}\n\nRaw Search Results:\n{search_results}")
    ])
    
    chain = prompt | llm
    summary = chain.invoke({
        "original_query": original_query,
        "sub_question": sub_question,
        "search_results": search_results
    })
    
    return {
        "research_data": [{
            "question": sub_question,
            "summary": summary.content,
            "raw_sources": search_results
        }]
    }
