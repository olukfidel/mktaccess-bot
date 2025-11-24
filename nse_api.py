import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nse_engine import NSEKnowledgeBase
import os

app = FastAPI(title="NSE Assistant API")

# Load API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

nse_engine = NSEKnowledgeBase(api_key)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"status": "running", "service": "NSE Assistant API"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    Endpoint to ask a question to the NSE bot.
    """
    try:
        # Note: Streaming is harder in basic REST, so we return full text for now
        # For streaming, you'd use StreamingResponse
        stream, sources = nse_engine.answer_question(request.query)
        
        full_response = ""
        if isinstance(stream, str):
            full_response = stream
        else:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
        
        return {
            "answer": full_response,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh")
def trigger_refresh():
    """
    Trigger a manual knowledge base update.
    """
    msg, logs = nse_engine.build_knowledge_base()
    return {"message": msg, "logs_count": len(logs)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)