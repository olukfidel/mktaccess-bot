import os
import requests
from bs4 import BeautifulSoup
import chromadb
from openai import OpenAI
import uuid
import urllib3

# Suppress SSL warnings so the logs stay clean
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class NSEKnowledgeBase:
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required")
        
        self.api_key = openai_api_key
        # Initialize OpenAI Client directly
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize ChromaDB (The memory)
        self.db_path = "./nse_db_pure"
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Create or get a collection for NSE data
        self.collection = self.chroma_client.get_or_create_collection(name="nse_data")

    def get_embedding(self, text):
        """Generates an embedding using OpenAI directly"""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

    def simple_text_splitter(self, text, chunk_size=1000, overlap=200):
        """A simple pure-python text splitter"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def scrape_nse_website(self, urls):
        """Scrapes text with stealth headers to bypass blocking"""
        # Real browser headers (Stealth Mode)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        logs = []
        scraped_data = []

        for url in urls:
            try:
                # verify=False bypasses SSL errors common with scrapers
                response = requests.get(url, headers=headers, timeout=15, verify=False)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Clean script/style tags
                    for item in soup(["script", "style", "nav", "footer"]):
                        item.decompose()
                    
                    text = soup.get_text(separator="\n")
                    
                    # Basic cleanup
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    clean_text = "\n".join(lines)
                    
                    # Only save if we actually got content
                    if len(clean_text) > 100:
                        scraped_data.append({"url": url, "text": clean_text})
                        logs.append(f"✅ Scraped: {url} ({len(clean_text)} chars)")
                    else:
                        logs.append(f"⚠️ Page empty or too short: {url}")
                else:
                    logs.append(f"❌ Failed {url}: Status {response.status_code}")
            except Exception as e:
                logs.append(f"❌ Error {url}: {str(e)}")
        
        return scraped_data, logs

    def build_knowledge_base(self, urls=None):
        """Scrapes and saves data to ChromaDB. CLEARS OLD DATA FIRST."""
        
        # --- NEW: Clear existing data to ensure freshness and no duplicates ---
        try:
            self.chroma_client.delete_collection("nse_data")
            self.collection = self.chroma_client.create_collection(name="nse_data")
        except Exception as e:
            # If collection didn't exist, just ignore
            pass
        # ---------------------------------------------------------------------

        if not urls:
            urls = [
                "https://www.nse.co.ke/",
                "https://www.nse.co.ke/market-statistics/",
                "https://www.nse.co.ke/market-statistics/daily-market-report/",
                "https://www.nse.co.ke/listed-companies/",
                "https://www.nse.co.ke/listed-company-announcements/",
                "https://www.nse.co.ke/derivatives/",
                "https://www.nse.co.ke/real-estate-investment-trusts/",
                "https://www.nse.co.ke/exchange-traded-funds/",
                "https://www.nse.co.ke/faqs/"
            ]

        # 1. Scrape
        data, logs = self.scrape_nse_website(urls)
        if not data:
            return "No data found to scrape.", logs

        # 2. Chunk and Embed
        count = 0
        for item in data:
            chunks = self.simple_text_splitter(item['text'])
            
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{"source": item['url']} for _ in chunks]
            
            # Generate embeddings for this batch
            embeddings = [self.get_embedding(chunk) for chunk in chunks]
            
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            count += len(chunks)

        return f"Success! Indexed {count} new text chunks.", logs

    def answer_question(self, query):
        """Retrieves data and asks GPT-3.5"""
        try:
            # 1. Embed user query
            query_embedding = self.get_embedding(query)
            
            # 2. Query Database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=4
            )
            
            # 3. Extract context
            if not results['documents'] or not results['documents'][0]:
                return "I couldn't find any relevant information in my knowledge base.", []

            retrieved_texts = results['documents'][0]
            sources = list(set([m['source'] for m in results['metadatas'][0]]))
            
            context_block = "\n---\n".join(retrieved_texts)

            # 4. Ask OpenAI
            system_prompt = (
                "You are an expert on the Nairobi Securities Exchange (NSE). "
                "Answer the question ONLY using the Context provided below. "
                "If the answer isn't in the context, say 'I don't have that information'."
                "\n\nContext:\n" + context_block
            )

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0
            )
            
            answer = response.choices[0].message.content
            return answer, sources

        except Exception as e:
            return f"Error: {str(e)}", []