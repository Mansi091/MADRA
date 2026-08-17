from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state import ResearchState

def writer_node(state: ResearchState) -> dict:
    print("WRITER NODE RUNNING")
    query = state.get("original_query")
    research_data = state.get("research_data", [])
    
    compiled_research = ""
    for idx, data in enumerate(research_data):
        compiled_research += f"\nSection: {data['question']}\n"
        compiled_research += f"Research Summary: {data['summary']}\n"
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert report writer. Write a comprehensive, well-structured markdown report that fully answers the user's query. Use ONLY the provided gathered research to write the report. Do not hallucinate external facts. Include clear headings and synthesize the information logically."),
        ("human", "User Query: {query}\n\nGathered Research Data:\n{compiled_research}")
    ])
    
    chain = prompt | llm
    report = chain.invoke({
        "query": query,
        "compiled_research": compiled_research
    })
    
    print("Report drafted successfully.")
    return {"draft_report": report.content, "final_report": report.content}
