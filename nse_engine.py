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
                "https://www.nse.co.ke/"
"https://www.nse.co.ke/home/"
"https://www.nse.co.ke/about-nse/"
"https://www.nse.co.ke/about-nse/history/"
"https://www.nse.co.ke/about-nse/vision-mission/"
"https://www.nse.co.ke/about-nse/board-of-directors/"
"https://www.nse.co.ke/about-nse/management-team/"
"https://www.nse.co.ke/listed-companies/"
"https://www.nse.co.ke/listed-companies/list/"
"https://www.nse.co.ke/share-price/"
"https://www.nse.co.ke/market-statistics/"
"https://www.nse.co.ke/market-statistics/daily-market-report/"
"https://www.nse.co.ke/market-statistics/weekly-market-report/"
"https://www.nse.co.ke/market-statistics/monthly-market-report/"
"https://www.nse.co.ke/data/"
"https://www.nse.co.ke/data/historical-data/"
"https://www.nse.co.ke/data/bond-data/"
"https://www.nse.co.ke/products/"
"https://www.nse.co.ke/products/equities/"
"https://www.nse.co.ke/products/derivatives/"
"https://www.nse.co.ke/products/reits/"
"https://www.nse.co.ke/products/etfs/"
"https://www.nse.co.ke/products/bonds/"
"https://www.nse.co.ke/usp/"
"https://www.nse.co.ke/ibuka/"
"https://www.nse.co.ke/clearing-settlement/"
"https://www.nse.co.ke/news/"
"https://www.nse.co.ke/news/announcements/"
"https://www.nse.co.ke/circulars/"
"https://www.nse.co.ke/media-center/"
"https://www.nse.co.ke/investor-education/"
"https://www.nse.co.ke/investor-relations/"
"https://www.nse.co.ke/careers/"
"https://www.nse.co.ke/contact-us/"
"https://www.nse.co.ke/faqs/"
"https://www.nse.co.ke/sustainability/"
"https://www.nse.co.ke/tenders/"
"https://www.nse.co.ke/procurement/"
"https://www.nse.co.ke/regulations/"
"https://www.nse.co.ke/trading-participants/"
"https://www.nse.co.ke/trading-participants/stockbrokers/"
"https://www.nse.co.ke/trading-participants/authorized-securities-dealers/"
"https://www.nse.co.ke/login/"
"https://www.nse.co.ke/online-trading-platform/"
"https://www.nse.co.ke/market-data-vendor/"
"https://live.nse.co.ke/"
"https://www.nse.co.ke/indices/"
"https://www.nse.co.ke/indices/nse-asi/"
"https://www.nse.co.ke/indices/nse-20/"
"https://www.nse.co.ke/indices/nse-25/"
"https://www.nse.co.ke/indices/nse-bond-index/"
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