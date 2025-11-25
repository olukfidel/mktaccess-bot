import uvicorn
import os
import logging
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
    else:
        logger.warning("CRITICAL: API keys not found in environment variables.")
    
    yield
    # Cleanup code (if needed) goes here
    logger.info("Shutting down NSE API.")

# --- App Definition ---
app = FastAPI(title="NSE Assistant API", lifespan=lifespan)

# Add CORS to allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

# --- Endpoints ---

@app.get("/")
def home():
    """Health check endpoint."""
    status = "ready" if nse_engine else "engine_failed"
    return {
        "status": "running", 
        "service": "NSE Assistant API", 
        "engine_status": status
    }

@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    Main endpoint to query the Knowledge Base.
    Note: Defined as 'def' (not async) to run in a threadpool 
    since the engine uses synchronous requests.
    """
    if not nse_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized. Check server logs.")

    try:
        # The engine returns a generator (stream) and a list of sources
        stream, sources = nse_engine.answer_question(request.query)
        
        full_response = ""
        
        # Consume the stream to build the full text response for JSON output
        # (If you wanted a streaming API response, you would use StreamingResponse here instead)
        if hasattr(stream, '__iter__') and not isinstance(stream, str):
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
        else:
            # Fallback if the engine returns a string directly
            full_response = str(stream)

        return {
            "answer": full_response,
            "sources": sources
        }

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh")
def trigger_refresh(background_tasks: BackgroundTasks):
    """Triggers a background crawl and re-index of the NSE data."""
    if not nse_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    def run_update_task():
        logger.info("Starting background knowledge base refresh...")
        try:
            msg, _ = nse_engine.build_knowledge_base()
            logger.info(f"Refresh complete: {msg}")
        except Exception as e:
            logger.error(f"Refresh failed: {e}")

    # Add the task to run in the background after returning the response
    background_tasks.add_task(run_update_task)
    
    return {"message": "Knowledge base refresh started in background."}

# --- Entry Point ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)