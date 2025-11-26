import uvicorn
import os
import logging
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nse_engine import NSEKnowledgeBase

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("NSE-API")

# --- Global State ---
nse_engine = None

# --- Lifespan Manager (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global nse_engine
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if api_key and pinecone_key:
        try:
            logger.info("Initializing NSE Knowledge Base...")
            nse_engine = NSEKnowledgeBase(api_key, pinecone_key)
            logger.info("NSE Engine Initialized Successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning("CRITICAL: API keys not found in environment variables. Queries will fail.")
    
    yield
    logger.info("Shutting down NSE API.")

# --- App Definition ---
app = FastAPI(title="NSE Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {
        "status": "running",
        "service": "NSE Assistant API",
        "engine_ready": nse_engine is not None,
        "backend": "Pinecone"
    }

@app.post("/ask")
def ask_question(request: QueryRequest):
    if not nse_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized (Check server logs)")
        
    try:
        stream, sources = nse_engine.answer_question(request.query)
        
        full_response = ""
        if hasattr(stream, '__iter__') and not isinstance(stream, str):
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
        else:
            full_response = str(stream)

        return {
            "answer": full_response,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh")
def trigger_refresh(background_tasks: BackgroundTasks):
    if not nse_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    def run_update_task():
        logger.info("Starting background knowledge base refresh...")
        try:
            msg, _ = nse_engine.build_knowledge_base()
            logger.info(f"Refresh complete: {msg}")
        except Exception as e:
            logger.error(f"Refresh failed: {e}")
            logger.error(traceback.format_exc())

    background_tasks.add_task(run_update_task)
    return {"message": "Knowledge base refresh started in background."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)