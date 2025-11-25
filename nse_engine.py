import os
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import uuid
import urllib3
import concurrent.futures
import time
import io
import re
import random
import datetime
import hashlib
from pypdf import PdfReader
from urllib.parse import urljoin, urlparse
from tenacity import retry, stop_after_attempt, wait_fixed
from rank_bm25 import BM25Okapi
from collections import defaultdict

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
MAX_CRAWL_DEPTH = 3
MAX_PAGES_TO_CRAWL = 1200
PINECONE_INDEX_NAME = "nse-data"
PINECONE_DIMENSION = 1536  # For text-embedding-3-small

class NSEKnowledgeBase:
    def __init__(self, openai_api_key, pinecone_api_key):
        if not openai_api_key or not pinecone_api_key:
            raise ValueError("API Keys are required")
        
        self.api_key = openai_api_key
        self.client = OpenAI(api_key=self.api_key)
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Ensure Index Exists
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"Creating Pinecone Index: {PINECONE_INDEX_NAME}...")
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(10) # Wait for init
            
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.session = requests.Session()

    # --- STATIC KNOWLEDGE ---
    def get_static_facts(self):
        return """
        [OFFICIAL_FACT_SHEET]
        TOPIC: NSE Leadership, Structure & Market Rules
        SOURCE: NSE Official Website / Annual Report 2025
        LAST_VERIFIED: November 2025

        CEO: Mr. Frank Mwiti (Appointed May 2, 2024)
        Chairman: Mr. Kiprono Kittony
        Location: The Exchange, 55 Westlands Road, Nairobi, Kenya
        Trading Hours: Mon-Fri, 09:30 am - 03:00 pm
        Currency: Kenyan Shilling (KES)
        Regulator: Capital Markets Authority (CMA)
        Depository: CDSC
        """

    # --- VECTOR OPERATIONS ---
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def get_embeddings_batch(self, texts):
        if not texts: return []
        sanitized = [t.replace("\n", " ") for t in texts]
        res = self.client.embeddings.create(input=sanitized, model=EMBEDDING_MODEL)
        return [d.embedding for d in res.data]

    def get_embedding(self, text):
        return self.get_embeddings_batch([text])[0]

    def build_knowledge_base(self):
        """Full crawl and index to Pinecone."""
        
        # Seed URLs (Your extensive list here)
        seeds = [
            "https://www.nse.co.ke/",
            "https://www.nse.co.ke/market-statistics/",
            "https://www.nse.co.ke/listed-companies/",
            "https://www.nse.co.ke/rules/",
            "https://www.nse.co.ke/guidelines/",
            "https://www.nse.co.ke/corporate-actions/",
            "https://www.nse.co.ke/listed-company-announcements/",
            "https://www.nse.co.ke/derivatives/",
            "https://www.nse.co.ke/nse-investor-calendar/",
            "https://www.nse.co.ke/policy-guidance-notes/",
            "https://www.nse.co.ke/circulars/",
            "https://www.nse.co.ke/how-to-become-a-trading-participant/",
            "https://www.nse.co.ke/e-digest/",
            "https://www.nse.co.ke/list-of-trading-participants/",
            "https://www.nse.co.ke/nse-events/",
            "https://www.nse.co.ke/csr/",
            "https://www.nse.co.ke/press-releases/",
            "https://www.nse.co.ke/publications/",
            "https://www.nse.co.ke/trading-participant-financials/",
            "https://www.nse.co.ke/faqs/"
        ]
        
        # Also prioritize specific PDFs you listed (Hardcoded list)
        hardcoded_pdfs = [
             "https://www.nse.co.ke/wp-content/uploads/Safaricom-PLC-Announcement-of-an-Interim-Dividend-For-The-Year-Ended-31-03-2025.pdf",
             "https://www.nse.co.ke/wp-content/uploads/NSE-2025-2029-Strategy.pdf",
             # ... (Add all other PDFs here)
        ]

        print("🕷️ Crawling NSE website...")
        found_pages, found_pdfs = self.crawl_site(seeds)
        all_urls = list(set(found_pages + found_pdfs + hardcoded_pdfs))
        
        print(f"📝 Found {len(all_urls)} total documents.")
        
        # Process & Upload
        total_chunks = self.scrape_and_upload(all_urls)
        
        return f"Knowledge Base Updated: {total_chunks} chunks uploaded to Pinecone.", []

    def scrape_and_upload(self, urls):
        """Scrapes URLs, chunks text, and uploads to Pinecone"""
        total_uploaded = 0
        batch_size = 50  # Pinecone batch limit recommended
        
        def process_url(url):
            try:
                res = self._fetch_url(url)
                if res.status_code != 200: return None
                
                ctype = "pdf" if url.lower().endswith(".pdf") or 'application/pdf' in res.headers.get('Content-Type', '') else "html"
                text = self._process_content(url, ctype, res.content)
                if not text: return None
                
                chunks = self.simple_text_splitter(text)
                if not chunks: return None
                
                vectors = []
                embeddings = self.get_embeddings_batch(chunks)
                
                for i, chunk in enumerate(chunks):
                    vector_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url + str(i)))
                    metadata = {
                        "text": chunk,
                        "source": url,
                        "date": datetime.date.today().isoformat(),
                        "type": ctype
                    }
                    vectors.append((vector_id, embeddings[i], metadata))
                
                return vectors
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return None

        # Parallel Processing
        vectors_to_upload = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_url, u): u for u in urls}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res: vectors_to_upload.extend(res)
        
        # Batch Upload to Pinecone
        # We upload in batches of 100 to be safe
        for i in range(0, len(vectors_to_upload), 100):
            batch = vectors_to_upload[i:i+100]
            self.index.upsert(vectors=batch)
            total_uploaded += len(batch)
            time.sleep(0.2) # Small delay to prevent rate limits
            
        return total_uploaded

    # --- RETRIEVAL ---
    def generate_context_queries(self, query):
        today = datetime.date.today().strftime("%Y-%m-%d")
        prompt = f"Generate 3 search queries for: '{query}'\nDate: {today}\n1. Keyword\n2. Concept\n3. Doc type\nOutput 3 lines."
        try:
            res = self.client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
            return [q.strip() for q in res.choices[0].message.content.split('\n') if q.strip()]
        except: return [query]

    def answer_question(self, query):
        # 1. Static Facts Injection
        context_text = self.get_static_facts() + "\n\n"
        visible_sources = set()
        
        try:
            # 2. Vector Search
            queries = self.generate_context_queries(query)
            # We use the first query (most specific) for the main vector search
            q_emb = self.get_embedding(queries[0])
            
            # Fetch top 15 matches
            results = self.index.query(vector=q_emb, top_k=15, include_metadata=True)
            
            # 3. Client-Side Re-Ranking (Hybrid)
            # We use BM25 on the retrieved chunks to boost exact keyword matches
            if results['matches']:
                docs = [m['metadata']['text'] for m in results['matches']]
                metas = [m['metadata'] for m in results['matches']]
                
                # Tokenize query for BM25
                tokenized_query = query.lower().split()
                tokenized_docs = [doc.lower().split() for doc in docs]
                
                bm25 = BM25Okapi(tokenized_docs)
                doc_scores = bm25.get_scores(tokenized_query)
                
                # Combine Vector Score + Keyword Score
                final_ranking = []
                for i, match in enumerate(results['matches']):
                    # Vector score is usually 0.7 - 0.9
                    # BM25 score can be 0 - 10+
                    # We normalize BM25 contribution
                    hybrid_score = match['score'] + (doc_scores[i] * 0.1)
                    
                    # Hard Rules Boosting
                    if "[OFFICIAL_FAQ]" in docs[i]: hybrid_score += 0.5
                    if "[OFFICIAL_FACT_SHEET]" in docs[i]: hybrid_score += 1.0
                    
                    final_ranking.append((hybrid_score, docs[i], metas[i]['source']))
                
                # Sort by new hybrid score
                final_ranking.sort(key=lambda x: x[0], reverse=True)
                
                # Take Top 5
                for _, text, source in final_ranking[:5]:
                    context_text += f"\n[Source: {source}]\n{text}\n---"
                    visible_sources.add(source)

        except Exception as e:
            print(f"Retrieval Error: {e}")
        
        # 4. Generation
        today = datetime.date.today().strftime("%Y-%m-%d")
        system_prompt = f"""You are the NSE Digital Assistant.
        TODAY: {today}
        RULES: 
        - Use [OFFICIAL_FACT_SHEET] for basics.
        - Prioritize [OFFICIAL_FAQ] content.
        - If unsure, say "I cannot find that specific info."
        CONTEXT: {context_text}"""

        stream = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            temperature=0,
            stream=True
        )
        return stream, list(visible_sources)

    # Helper methods (Crawling logic same as before)
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_url(self, url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        return self.session.get(url, headers=headers, verify=False, timeout=10)

    def crawl_site(self, seed_urls):
        # (Same crawler logic as previous file - keeps it robust)
        visited = set()
        to_visit = set(seed_urls)
        found_pages = set()
        found_pdfs = set()
        
        count = 0
        while to_visit and count < MAX_PAGES_TO_CRAWL:
            try: url = to_visit.pop()
            except: break
            if url in visited: continue
            visited.add(url)
            
            if "nse.co.ke" not in url: continue
            
            try:
                res = self._fetch_url(url)
                if res.status_code == 200:
                    if url.endswith(".pdf") or 'pdf' in res.headers.get('Content-Type', ''):
                        found_pdfs.add(url)
                    else:
                        found_pages.add(url)
                        soup = BeautifulSoup(res.content, 'html.parser')
                        for link in soup.find_all('a', href=True):
                            full = urljoin(url, link['href'])
                            if "nse.co.ke" in full and full not in visited:
                                to_visit.add(full)
                count += 1
            except: pass
        return list(found_pages), list(found_pdfs)

    def _extract_text_from_pdf(self, pdf_bytes):
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except:
            return ""

    def _process_content(self, url, ctype, content):
        # (Same tagging logic as before)
        tag = "[GENERAL]"
        if "statistics" in url: tag = "[MARKET_DATA]"
        # ... add all your tags here ...
        
        if ctype == "pdf":
            text = self._extract_text_from_pdf(content)
        else:
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator="\n")
            
        return f"{tag} SOURCE: {url}\n\n{text.strip()}"

    def clean_text_chunk(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    def simple_text_splitter(self, text, chunk_size=1000, overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - overlap
        return chunks