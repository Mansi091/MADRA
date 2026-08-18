from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import uuid
import os
from graph import graph

app = FastAPI(title="Deep Research Agent API")

# Mount the static directory to serve index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    query = req.query
    
    async def event_generator():
        initial_state = {
            "original_query": query,
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
        
        try:
            import time
            node_start_time = time.time()
            # 1. Tell client the first node is starting
            yield f"data: {json.dumps({'type': 'tool_start', 'name': 'Planner'})}\n\n"
            
            # 2. Stream the graph up to the interrupt
            for event in graph.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    # Node finished
                    duration = time.time() - node_start_time
                    data = {
                        'type': 'tool_end', 
                        'name': node_name.capitalize(), 
                        'status': 'success',
                        'duration': round(duration, 1)
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    
                    # Predict next node to yield 'tool_start'
                    if node_name == 'planner':
                        node_start_time = time.time()
                        yield f"data: {json.dumps({'type': 'tool_start', 'name': 'Researcher'})}\n\n"
                    elif node_name == 'researcher':
                        node_start_time = time.time()
                        yield f"data: {json.dumps({'type': 'tool_start', 'name': 'Reflector'})}\n\n"
                    elif node_name == 'reflector':
                        current_state = graph.get_state(config)
                        next_node = current_state.next
                        if next_node and len(next_node) > 0:
                            node_start_time = time.time()
                            yield f"data: {json.dumps({'type': 'tool_start', 'name': next_node[0].capitalize()})}\n\n"
                            
            # 3. Extract internal score at the pause
            current_state = graph.get_state(config)
            score = current_state.values.get("research_score", 0)
            
            # 4. Resume the graph past the interrupt (Writer node)
            node_start_time = time.time()
            yield f"data: {json.dumps({'type': 'tool_start', 'name': 'Writer'})}\n\n"
            
            for event in graph.stream(None, config=config, stream_mode="updates"):
                 for node_name, node_state in event.items():
                    duration = time.time() - node_start_time
                    data = {
                        'type': 'tool_end', 
                        'name': node_name.capitalize(), 
                        'status': 'success',
                        'duration': round(duration, 1)
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    
            # 5. Fetch the final report
            final_state = graph.get_state(config).values
            final_report = final_state.get('draft_report', 'Failed to generate report.')
            
            # 5. Yield final response
            done_data = {
                'type': 'done', 
                'report': final_report, 
                'score': score
            }
            yield f"data: {json.dumps(done_data)}\n\n"
            
        except Exception as e:
            error_data = {'type': 'error', 'message': str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    # Return the generator as a StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
