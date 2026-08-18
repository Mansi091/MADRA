from langchain_groq import ChatGroq
from state import ResearchState

def writer_node(state: ResearchState) -> dict:
    print("\n[WRITER] GENERATING CITED REPORT")
    query = state.get("original_query")
    evidence_list = state.get("evidence", [])
    critique = state.get("report_critique", "")
    draft_report = state.get("draft_report", "")
    
    compiled_evidence = ""
    # Map sources to sequential IDs
    source_mapping = {}
    for i, ev in enumerate(evidence_list):
        source_id = i + 1
        source_mapping[source_id] = ev['url']
        compiled_evidence += f"\nSource [{source_id}] ({ev['url']}):\n"
        for claim in ev.get("claims", []):
            compiled_evidence += f"  - {claim}\n"
            
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
    if critique:
        print("  -> Refining based on critique.")
        prompt = f"""
        You are an expert report writer. Revise your draft report based on the editor's critique.
        
        Original Query: {query}
        
        Editor's Critique:
        {critique}
        
        Previous Draft:
        {draft_report}
        
        Verified Evidence:
        {compiled_evidence}
        
        CRITICAL INSTRUCTIONS:
        1. Fully address all points in the critique.
        2. Every factual claim MUST include its inline footnote citation, e.g. [1] or [2], based on the Verified Evidence provided.
        3. At the very end of the report, you MUST include a '## References' section listing all used Source IDs and their URLs exactly as provided.
        """
    else:
        print("  -> Drafting initial report.")
        prompt = f"""
        You are an expert report writer. Write a comprehensive report answering the following query.
        
        Original Query: {query}
        
        You MUST use the provided evidence. 
        CRITICAL INSTRUCTIONS:
        1. Every factual claim you make MUST include an inline footnote citation, e.g. [1] or [2], corresponding to the Source ID.
        2. Do not introduce facts that are not present in the verified evidence.
        3. If evidence is conflicting, explicitly mention the conflict.
        4. At the very end of the report, you MUST include a '## References' section listing all used Source IDs and their URLs exactly as provided.
        
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
