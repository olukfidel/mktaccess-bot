import os
import requests
from bs4 import BeautifulSoup
import chromadb
from openai import OpenAI
import uuid

class NSEKnowledgeBase:
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required")
        
        self.api_key = openai_api_key
        # Initialize OpenAI Client directly
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize ChromaDB (The memory)
        # We use a persistent folder so data is saved
        self.db_path = "./nse_db_pure"
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Create or get a collection (like a table) for NSE data
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
            # Move forward by chunk_size minus overlap
            start += chunk_size - overlap
        return chunks

    def scrape_nse_website(self, urls):
        """Scrapes text from URLs"""
        headers = {'User-Agent': 'Mozilla/5.0'}
        logs = []
        scraped_data = []

        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Clean script/style tags
                    for item in soup(["script", "style", "nav", "footer"]):
                        item.decompose()
                    
                    text = soup.get_text(separator="\n")
                    # Basic cleanup
                    clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
                    
                    scraped_data.append({"url": url, "text": clean_text})
                    logs.append(f"✅ Scraped: {url}")
                else:
                    logs.append(f"❌ Failed {url}: {response.status_code}")
            except Exception as e:
                logs.append(f"❌ Error {url}: {str(e)}")
        
        return scraped_data, logs

    def build_knowledge_base(self, urls=None):
        """Scrapes and saves data to ChromaDB"""
        if not urls:
            urls = [
                "https://www.nse.co.ke/",
                "https://www.nse.co.ke/market-statistics/",
                "https://www.nse.co.ke/listed-companies/",
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
            # Note: In a production app we'd batch this API call, 
            # but looping is fine for small updates.
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
            # Chroma returns lists of lists
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