from langchain_groq import ChatGroq
from state import ResearchState

def critic_node(state: ResearchState) -> dict:
    print("\n[CRITIC] REVIEWING DRAFT REPORT")
    draft = state.get("draft_report", "")
    query = state.get("original_query", "")
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    
    prompt = f"""
    You are an expert editor and fact-checker. Review the following draft report.
    
    Original Query: {query}
    
    Draft Report:
    {draft}
    
    Critique the report based on the following criteria:
    1. Clarity and Structure
    2. Factual consistency and missing gaps
    3. Proper usage of inline citations (e.g. [1], [2])
    4. Redundancy
    
    Provide specific, actionable revision suggestions for the writer to improve the report in its final rewrite.
    Do NOT rewrite the report yourself. Only output the critique.
    """
    
    response = llm.invoke(prompt)
    critique = response.content
    print("  -> Critique generated successfully.")
    
    current_revisions = state.get("revision_iterations", 0)
    
    return {
        "report_critique": critique,
        "revision_iterations": current_revisions + 1
    }
