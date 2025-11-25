import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from nse_engine import NSEKnowledgeBase
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NSE-API")

app = FastAPI(title="NSE Assistant API")

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    try:
        nse_engine = NSEKnowledgeBase(api_key)
        logger.info("NSE Engine Initialized Successfully")
    except Exception as e:
        logger.error(f"Failed to init engine: {e}")
        nse_engine = None
else:
    logger.warning("OPENAI_API_KEY not found. API will start but queries will fail.")
    nse_engine = None

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"status": "running", "service": "NSE Assistant API", "engine_ready": nse_engine is not None}

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not nse_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
        
    try:
        stream, sources = nse_engine.answer_question(request.query)
        full_response = ""
        if isinstance(stream, str):
            full_response = stream
        else:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
        
        return {"answer": full_response, "sources": sources}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh")
def trigger_refresh(background_tasks: BackgroundTasks):
    if not nse_engine:
         raise HTTPException(status_code=503, detail="Engine not initialized")
    
    def run_update():
        logger.info("Starting background refresh...")
        try:
            msg, _ = nse_engine.build_knowledge_base()
            logger.info(f"Refresh complete: {msg}")
        except Exception as e:
            logger.error(f"Refresh failed: {e}")

    background_tasks.add_task(run_update)
    return {"message": "Refresh started in background"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)