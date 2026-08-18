import uuid
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from state import ResearchState
from nodes.planner import planner_node
from nodes.researcher import researcher_node
from nodes.reflector import reflector_node
from nodes.writer import writer_node
from nodes.critic import critic_node

def map_researchers(state: ResearchState):
    return [
        Send("researcher", {"sub_question": q, "original_query": state.get("original_query")})
        for q in state.get("sub_questions", [])
    ]

def should_continue(state: ResearchState):
    score = state.get("research_score", 0.0)
    iterations = state.get("research_iterations", 0)
    
    print(f"ROUTING: Score is {score}, Iteration is {iterations}")
    
    if score >= 8.0 or iterations >= 3:
        return "writer"
    else:
        return "planner"

def should_finalize(state: ResearchState):
    if state.get("revision_iterations", 0) >= 1:
        return END
    else:
        return "writer"

builder = StateGraph(ResearchState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("reflector", reflector_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", map_researchers, ["researcher"])
builder.add_edge("researcher", "reflector")
builder.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "planner": "planner",
        "writer": "writer"
    }
)
builder.add_edge("writer", "critic")
builder.add_conditional_edges(
    "critic",
    should_finalize,
    {
        "writer": "writer",
        END: END
    }
)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory, interrupt_before=["writer"])

if __name__ == "__main__":
    print("    DEEP RESEARCH AGENT STARTING    ")
    
    user_query = input("Enter your query: ")
    
    if not user_query.strip():
        user_query = "Compare the impact of AI on the healthcare industry in India from 2020 to 2026."
        print(f"No query provided. Using default:\n'{user_query}'\n")
        
    initial_state = {
        "original_query": user_query,
        "sub_questions": [],
        "research_data": [],
        "research_score": 0.0,
        "evaluation_feedback": "",
        "research_iterations": 0,
        "draft_report": "",
        "final_report": ""
    }
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Running graph with STREAMING (will pause for human approval before writing)...")
    
    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_state in event.items():
            print(f"\n[STREAM] Finished running node: {node_name}")
    
    print("\nHuman in the loop ")
    current_state = graph.get_state(config)
    print(f"Next node to run: {current_state.next}")
    print(f"Current Research Score: {current_state.values.get('research_score')}")
    
    user_input = input("\nPress Enter to approve the research and start writing the report: ")
    
    print("\nRESUMING GRAPH WITH STREAMING")
    for event in graph.stream(None, config=config, stream_mode="updates"):
        for node_name, node_state in event.items():
            print(f"\n[STREAM] Finished running node: {node_name}")
            
    final_state = graph.get_state(config).values
    
    print("\nFINAL STATE")
    print(f"Iterations run: {final_state.get('research_iterations')}")
    print(f"Final Report Draft:\n{final_state.get('draft_report')}")
