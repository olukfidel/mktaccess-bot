import os
import requests
from bs4 import BeautifulSoup
import chromadb
from openai import OpenAI
import uuid
import urllib3
import concurrent.futures
import time
import io
from pypdf import PdfReader # The tool to read PDFs
from urllib.parse import urljoin # To fix relative links

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
        return self.collection.count() > 0

    def get_last_update_time(self):
        try:
            if os.path.exists("last_update.txt"):
                with open("last_update.txt", "r") as f:
                    return float(f.read().strip())
        except:
            return 0.0
        return 0.0

    def get_embeddings_batch(self, texts):
        if not texts: return []
        sanitized_texts = [t.replace("\n", " ") for t in texts]
        try:
            response = self.client.embeddings.create(input=sanitized_texts, model="text-embedding-3-small")
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Embedding Error: {e}")
            return []

    def get_embedding(self, text):
        return self.get_embeddings_batch([text])[0]

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

    def _extract_text_from_pdf(self, pdf_content):
        """Helper to extract text from PDF binary data"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception:
            return ""

    def _scrape_single_url(self, url):
        """Scrapes HTML OR PDF based on content type"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Upgrade-Insecure-Requests': '1',
        }
        found_pdfs = [] # List to store PDF links found on this page
        
        try:
            response = requests.get(url, headers=headers, timeout=15, verify=False)
            
            # 1. Handle PDF Files directly
            if url.lower().endswith(".pdf") or 'application/pdf' in response.headers.get('Content-Type', ''):
                pdf_text = self._extract_text_from_pdf(response.content)
                if len(pdf_text) > 100:
                    clean_text = "SOURCE PDF: " + url + "\n\n" + pdf_text
                    return {"url": url, "text": clean_text}, f"📄 PDF Parsed: {url}", []
                else:
                    return None, f"⚠️ Empty PDF: {url}", []

            # 2. Handle HTML Files
            elif response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # A. Hunt for PDF links on this page
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.lower().endswith('.pdf'):
                        full_pdf_url = urljoin(url, href)
                        found_pdfs.append(full_pdf_url)

                # B. Clean and Extract HTML Text
                for item in soup(["script", "style", "nav", "footer"]):
                    item.decompose()
                text = soup.get_text(separator="\n")
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                clean_text = "\n".join(lines)
                
                if len(clean_text) > 100:
                    return {"url": url, "text": clean_text}, f"✅ Scraped: {url}", found_pdfs
                else:
                    return None, f"⚠️ Empty: {url}", found_pdfs
            else:
                return None, f"❌ Failed {url}: {response.status_code}", []
                
        except Exception as e:
            return None, f"❌ Error {url}: {str(e)}", []

    def scrape_nse_website(self, urls):
        scraped_data = []
        logs = []
        discovered_pdfs = set()

        # Phase 1: Scrape Main URLs + Find PDF Links
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self._scrape_single_url, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                data, log_msg, pdfs_found = future.result()
                logs.append(log_msg)
                if data:
                    scraped_data.append(data)
                # Collect PDF links found on the page
                for pdf in pdfs_found:
                    discovered_pdfs.add(pdf)

        # Phase 2: Scrape the Discovered PDFs
        # Limit to 15 PDFs to prevent overload during demo
        pdf_list = list(discovered_pdfs)[:15] 
        if pdf_list:
            logs.append(f"🔍 Found {len(discovered_pdfs)} PDFs. Scraping top 15...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(self._scrape_single_url, url): url for url in pdf_list}
                for future in concurrent.futures.as_completed(future_to_url):
                    data, log_msg, _ = future.result()
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
                "https://www.nse.co.ke/", "https://www.nse.co.ke/market-statistics/", 
                "https://www.nse.co.ke/listed-companies/", "https://www.nse.co.ke/market-statistics/daily-market-report/",
                "https://www.nse.co.ke/products/equities/", "https://www.nse.co.ke/products/derivatives/",
                "https://www.nse.co.ke/products/reits/", "https://www.nse.co.ke/products/etfs/",
                "https://www.nse.co.ke/products/bonds/", "https://www.nse.co.ke/news/announcements/",
                "https://www.nse.co.ke/faqs/", "https://www.nse.co.ke/rules/"
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
        
        try:
            with open("last_update.txt", "w") as f:
                f.write(str(time.time()))
        except:
            pass

        return f"Refresh Complete! Indexed {total_chunks} chunks (HTML + PDF).", logs

    def generate_context_queries(self, original_query):
        system_prompt = (
            "You are a search assistant for the Nairobi Securities Exchange. "
            "Generate 3 different search queries to find the answer in a database.\n"
            "1. Literal keywords.\n"
            "2. Definition or concept.\n"
            "3. Related company or report data.\n"
            "Output ONLY the 3 queries, one per line."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": original_query}],
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
            
            results = self.collection.query(query_embeddings=query_embeddings, n_results=5)
            
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
                "The context includes website text AND PDF reports. "
                "Synthesize the information to answer the user's specific intent. "
                "\n\n"
                "Rules:\n"
                "1. Prioritize definitions for 'what is' questions.\n"
                "2. Prioritize data/reports for 'price' or 'stat' questions.\n"
                "3. Always cite the source URL (especially if it's a PDF)."
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