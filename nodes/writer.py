from langchain_groq import ChatGroq
from state import ResearchState

def writer_node(state: ResearchState) -> dict:
    print("\n[WRITER] GENERATING CITED REPORT")
    query = state.get("original_query")
    evidence_list = state.get("evidence", [])
    
    compiled_evidence = ""
    for ev in evidence_list:
        compiled_evidence += f"\nSource [{ev['source_id']}] ({ev['url']}):\n"
        for claim in ev.get("claims", []):
            compiled_evidence += f"  - {claim}\n"
            
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
    prompt = f"""
    You are an expert report writer. Write a comprehensive report answering the following query.
    
    Original Query: {query}
    
    You MUST use the provided evidence. 
    Every factual claim you make MUST include its citation ID in brackets, e.g. [a1b2c3d4].
    Do not introduce facts that are not present in the verified evidence.
    If evidence is conflicting, explicitly mention the conflict.
    
    At the end of the report, include a 'References' section listing the Source IDs and their URLs.
    
    Verified Evidence:
    {compiled_evidence}
    """
    
    response = llm.invoke(prompt)
    print("  -> Report generated successfully.")
    
    report_text = response.content
    if isinstance(report_text, list):
        report_text = "".join([chunk.get("text", "") for chunk in report_text if isinstance(chunk, dict)])
    elif not isinstance(report_text, str):
        report_text = str(report_text)
    
    return {
        "draft_report": report_text,
        "final_report": report_text
    }
