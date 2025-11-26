import uvicorn
import os
import logging
import traceback
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Don't import engine at top level to prevent blocking import
# from nse_engine import NSEKnowledgeBase 

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("NSE-API")

# --- Global State ---
nse_engine = None
engine_loading = False

# --- Lifespan Manager (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # We start the initialization in a non-blocking way
    asyncio.create_task(init_engine())
    yield
    logger.info("Shutting down NSE API.")

async def init_engine():
    global nse_engine, engine_loading
    if nse_engine is not None or engine_loading:
        return

    engine_loading = True
    logger.info("Starting background engine initialization...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if api_key and pinecone_key:
        try:
            # Import here to avoid top-level blocking
            from nse_engine import NSEKnowledgeBase
            # Run blocking init in a threadpool
            nse_engine = await asyncio.to_thread(NSEKnowledgeBase, api_key, pinecone_key)
            logger.info("NSE Engine Initialized Successfully (Background).")
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning("CRITICAL: API keys missing. Queries will fail.")
    
    engine_loading = False

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
    """Health check endpoint - responds immediately."""
    status = "initializing" if engine_loading else ("ready" if nse_engine else "failed")
    return {
        "status": "running",
        "engine_status": status,
        "service": "NSE Assistant API"
    }

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not nse_engine:
        if engine_loading:
            raise HTTPException(status_code=503, detail="System is warming up. Please try again in 30 seconds.")
        raise HTTPException(status_code=503, detail="Engine failed to initialize. Check server logs.")
        
    try:
        # Offload heavy blocking logic to thread
        response_data = await asyncio.to_thread(get_answer_sync, request.query)
        return response_data

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_answer_sync(query):
    stream, sources = nse_engine.answer_question(query)
    full_response = ""
    if hasattr(stream, '__iter__') and not isinstance(stream, str):
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
    else:
        full_response = str(stream)
    return {"answer": full_response, "sources": sources}

@app.post("/refresh")
async def trigger_refresh(background_tasks: BackgroundTasks):
    if not nse_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    def run_update_task():
        logger.info("Starting background refresh...")
        try:
            msg, _ = nse_engine.build_knowledge_base()
            logger.info(f"Refresh complete: {msg}")
        except Exception as e:
            logger.error(f"Refresh failed: {e}")

    background_tasks.add_task(run_update_task)
    return {"message": "Refresh started in background."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)