import os
from nse_engine import NSEKnowledgeBase
from dotenv import load_dotenv

# Load environment variables from .env file if you have one, 
# or ensure they are set in your terminal session.
load_dotenv()

def main():
    # 1. Get API Keys
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not openai_key or not pinecone_key:
        print("Error: Please set OPENAI_API_KEY and PINECONE_API_KEY environment variables.")
        return

    print("🚀 Initializing NSE Engine...")
    try:
        # Initialize the engine
        # This will connect to Pinecone and OpenAI
        engine = NSEKnowledgeBase(openai_api_key=openai_key, pinecone_api_key=pinecone_key)
        
        print("🕷️ Starting the scraping and indexing process...")
        print("This may take a few minutes. Please wait...")
        
        # Trigger the build process
        status_message, logs = engine.build_knowledge_base()
        
        print("\n✅ Process Completed!")
        print(f"Status: {status_message}")
        
        # Optional: Print logs if you want to see details
        # print("\n--- Detailed Logs ---")
        # for log in logs:
        #     print(log)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()