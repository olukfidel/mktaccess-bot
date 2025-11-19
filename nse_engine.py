import os
import requests
from bs4 import BeautifulSoup
import chromadb
from openai import OpenAI
import uuid
import urllib3
import concurrent.futures
import time

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class NSEKnowledgeBase:
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required")
        
        self.api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        
        self.db_path = "./nse_db_pure"
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="nse_data")

    def has_data(self):
        """Checks if the database has any data at all"""
        return self.collection.count() > 0

    def get_last_update_time(self):
        """Returns the timestamp of the last successful scrape"""
        try:
            if os.path.exists("last_update.txt"):
                with open("last_update.txt", "r") as f:
                    return float(f.read().strip())
        except:
            return 0.0
        return 0.0

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model="text-embedding-3-small").data[0].embedding

    def get_embeddings_batch(self, texts):
        if not texts: return []
        sanitized_texts = [t.replace("\n", " ") for t in texts]
        response = self.client.embeddings.create(input=sanitized_texts, model="text-embedding-3-small")
        return [data.embedding for data in response.data]

    def simple_text_splitter(self, text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def _scrape_single_url(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Upgrade-Insecure-Requests': '1',
        }
        try:
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                for item in soup(["script", "style", "nav", "footer"]):
                    item.decompose()
                text = soup.get_text(separator="\n")
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                clean_text = "\n".join(lines)
                if len(clean_text) > 100:
                    return {"url": url, "text": clean_text}, f"✅ Scraped: {url}"
                else:
                    return None, f"⚠️ Empty: {url}"
            else:
                return None, f"❌ Failed {url}: {response.status_code}"
        except Exception as e:
            return None, f"❌ Error {url}: {str(e)}"

    def scrape_nse_website(self, urls):
        scraped_data = []
        logs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self._scrape_single_url, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                data, log_msg = future.result()
                logs.append(log_msg)
                if data:
                    scraped_data.append(data)
        return scraped_data, logs

    def build_knowledge_base(self, urls=None):
        try:
            self.chroma_client.delete_collection("nse_data")
            self.collection = self.chroma_client.create_collection(name="nse_data")
        except:
            pass

        if not urls:
            urls = [
                "https://www.nse.co.ke/",
"https://www.nse.co.ke/home/",
"https://www.nse.co.ke/about-nse/",
"https://www.nse.co.ke/about-nse/history/",
"https://www.nse.co.ke/about-nse/vision-mission/",
"https://www.nse.co.ke/about-nse/board-of-directors/",
"https://www.nse.co.ke/about-nse/management-team/",
"https://www.nse.co.ke/listed-companies/",
"https://www.nse.co.ke/listed-companies/list/",
"https://www.nse.co.ke/share-price/",
"https://www.nse.co.ke/market-statistics/",
"https://www.nse.co.ke/market-statistics/daily-market-report/",
"https://www.nse.co.ke/market-statistics/weekly-market-report/",
"https://www.nse.co.ke/market-statistics/monthly-market-report/",
"https://www.nse.co.ke/data/",
"https://www.nse.co.ke/data/historical-data/",
"https://www.nse.co.ke/data/bond-data/",
"https://www.nse.co.ke/products/",
"https://www.nse.co.ke/products/equities/",
"https://www.nse.co.ke/products/derivatives/",
"https://www.nse.co.ke/products/reits/",
"https://www.nse.co.ke/products/etfs/",
"https://www.nse.co.ke/products/bonds/",
"https://www.nse.co.ke/usp/",
"https://www.nse.co.ke/ibuka/",
"https://www.nse.co.ke/clearing-settlement/",
"https://www.nse.co.ke/news/",
"https://www.nse.co.ke/news/announcements/",
"https://www.nse.co.ke/circulars/",
"https://www.nse.co.ke/media-center/",
"https://www.nse.co.ke/investor-education/",
"https://www.nse.co.ke/investor-relations/",
"https://www.nse.co.ke/careers/",
"https://www.nse.co.ke/contact-us/",
"https://www.nse.co.ke/faqs/",
"https://www.nse.co.ke/sustainability/",
"https://www.nse.co.ke/tenders/",
"https://www.nse.co.ke/procurement/",
"https://www.nse.co.ke/regulations/",
"https://www.nse.co.ke/trading-participants/",
"https://www.nse.co.ke/trading-participants/stockbrokers/",
"https://www.nse.co.ke/trading-participants/authorized-securities-dealers/",
"https://www.nse.co.ke/login/",
"https://www.nse.co.ke/online-trading-platform/",
"https://www.nse.co.ke/market-data-vendor/",
"https://live.nse.co.ke/",
"https://www.nse.co.ke/indices/",
"https://www.nse.co.ke/indices/nse-asi/",
"https://www.nse.co.ke/indices/nse-20/",
"https://www.nse.co.ke/indices/nse-25/",
"https://www.nse.co.ke/indices/nse-bond-index/"
            ]

        data, logs = self.scrape_nse_website(urls)
        if not data:
            return "No data found.", logs

        total_chunks = 0
        all_chunks = []
        all_ids = []
        all_metadatas = []

        for item in data:
            chunks = self.simple_text_splitter(item['text'])
            for chunk in chunks:
                all_chunks.append(chunk)
                all_ids.append(str(uuid.uuid4()))
                all_metadatas.append({"source": item['url']})

        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i : i + batch_size]
            batch_ids = all_ids[i : i + batch_size]
            batch_metas = all_metadatas[i : i + batch_size]
            batch_embeddings = self.get_embeddings_batch(batch_chunks)
            
            if batch_embeddings:
                self.collection.add(
                    documents=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                total_chunks += len(batch_chunks)
        
        # Save the current timestamp as the last update time
        try:
            with open("last_update.txt", "w") as f:
                f.write(str(time.time()))
        except:
            pass

        return f"Refresh Complete! Indexed {total_chunks} chunks.", logs

    def generate_context_queries(self, original_query):
        """Uses AI to expand the single user query into multiple specific search terms"""
        system_prompt = (
            "You are a search assistant for the Nairobi Securities Exchange. "
            "The user asked a question. Generate 3 different search queries to find the answer "
            "in a document database.\n"
            "1. One query for the literal keywords.\n"
            "2. One query for the DEFINITION or CONCEPT (if applicable).\n"
            "3. One query for specific COMPANY or DATA (if applicable).\n"
            "Output ONLY the 3 queries, one per line."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": original_query}
                ],
                temperature=0.5
            )
            queries = [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()]
            return queries[:3]
        except:
            return [original_query]

    def answer_question(self, query):
        try:
            search_queries = self.generate_context_queries(query)
            query_embeddings = self.get_embeddings_batch(search_queries)
            
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=5 
            )
            
            unique_texts = set()
            sources = set()
            
            for i, doc_list in enumerate(results['documents']):
                meta_list = results['metadatas'][i]
                for j, text in enumerate(doc_list):
                    if text and text not in unique_texts:
                        unique_texts.add(text)
                        sources.add(meta_list[j]['source'])
            
            if not unique_texts:
                return "I couldn't find any relevant information in my knowledge base.", []

            context_block = "\n---\n".join(list(unique_texts))

            system_prompt = (
                "You are an expert on the Nairobi Securities Exchange (NSE). "
                "Answer the question using the Context provided below. "
                "The context may contain definitions, company data, or rules. "
                "Synthesize the information to answer the user's specific intent. "
                "\n\n"
                "Rules:\n"
                "1. If the user asks for a definition (e.g. 'what is'), prioritize the definition chunks over company data.\n"
                "2. If the user asks for data (e.g. 'price of'), prioritize the company data chunks.\n"
                "3. Always cite the source URL."
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
            return response.choices[0].message.content, list(sources)

        except Exception as e:
            return f"Error: {str(e)}", []