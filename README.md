📈 NSE Digital Assistant (Enterprise Edition)

An AI-powered financial assistant for the Nairobi Securities Exchange (NSE). This application helps investors find share prices, analyze market reports, and understand trading rules using a robust RAG (Retrieval Augmented Generation) engine.

🏗️ Architecture

This project uses a Decoupled Architecture for scalability and performance:

Backend (Brain): Python FastAPI application hosted on Railway.
Handles data scraping (HTML + PDF parsing).
Manages Vector Database (Pinecone).
Generates answers using OpenAI GPT-4o-mini.
Frontend (Face): Next.js (React) application hosted on Vercel.
Provides a responsive, chat-like interface.
Connects to the backend via REST API.

🚀 1. Backend Setup (Railway)

The backend logic resides in nse_api.py and nse_engine.py.

Deployment Steps

Push this repository to GitHub (excluding node_modules and nse-frontend folder if separate).

Log in to Railway.
Create a New Project > Deploy from GitHub.
Go to Settings > Variables and add:

OPENAI_API_KEY: Your OpenAI Key (sk-...)

PINECONE_API_KEY: Your Pinecone Key.

Railway will detect the Python app. Ensure the Start Command is:
uvicorn nse_api:app --host 0.0.0.0 --port $PORT
Once deployed, copy your Public URL (e.g., https://your-app.up.railway.app).

🌐 2. Frontend Setup (Vercel)

The frontend code is located in the nse-frontend/ directory (if you followed the setup).

Local Development

Navigate to the frontend folder: cd nse-frontend

Install dependencies: npm install

Create .env.local:

NEXT_PUBLIC_API_URL=http://localhost:8000  # Or your Railway URL


Run locally: npm run dev

Deployment Steps

Push your Next.js code to a GitHub repository.

Log in to Vercel.

Add New Project and select your repo.

In Environment Variables, add:

NEXT_PUBLIC_API_URL: Your Railway Backend URL (e.g., https://your-app.up.railway.app).

Click Deploy.

🧠 Knowledge Base Features

Hybrid Search: Combines semantic search (vectors) with keyword matching for high accuracy.

Smart Crawling: Recursively crawls NSE website pages and downloads key PDFs (Financial Results, Trading Rules).
Auto-Updates: Checks for stale data (>24 hours) and refreshes automatically in the background.

Fact Sheet: Hardcoded high-priority facts (CEO, Location) are injected into every prompt to prevent hallucinations on basic info.

🛠️ Tech Stack

Backend: FastAPI, Uvicorn, Tenacity, BeautifulSoup4, PyPDF, Rank_BM25.

AI/DB: OpenAI (GPT-4o-mini, text-embedding-3-small), Pinecone (Serverless).

Frontend: Next.js 14, Tailwind CSS, Axios, Lucide React.

© 2025 Market Access Bot