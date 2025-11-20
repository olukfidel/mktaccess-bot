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
import re
import datetime
from pypdf import PdfReader
from urllib.parse import urljoin

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
        
        try:
            self.collection = self.chroma_client.get_or_create_collection(name="nse_data")
        except Exception:
            self.collection = None

    def has_data(self):
        try:
            if self.collection is None: return False
            return self.collection.count() > 0
        except:
            return False

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
            # Ensure we use the correct model for embeddings
            response = self.client.embeddings.create(input=sanitized_texts, model="text-embedding-3-small")
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Embedding Error: {e}")
            return []

    def get_embedding(self, text):
        return self.get_embeddings_batch([text])[0]

    def clean_text_chunk(self, text):
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def simple_text_splitter(self, text, chunk_size=1500, overlap=300):
        # Improved splitter that tries not to break mid-sentence
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            
            # If we are not at the end of text, try to find the last period to break cleanly
            if end < text_len:
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + (chunk_size // 2):
                    end = last_period + 1 # Include the period
            
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap # Move window forward
            
            # Ensure we don't get stuck
            if start >= end:
                start = end
                
        return chunks

    def _extract_text_from_pdf(self, pdf_content):
        try:
            pdf_file = io.BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                try:
                    # layout mode preserves table structure better
                    page_text = page.extract_text(extraction_mode="layout")
                except:
                    page_text = page.extract_text()
                text += page_text + "\n"
            return text
        except Exception:
            return ""

    def _scrape_single_url(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # SMART TAGGING based on URL patterns
        tag = "[GENERAL]"
        if "statistics" in url: tag = "[MARKET_DATA]"
        elif "management" in url or "directors" in url: tag = "[LEADERSHIP]"
        elif "contact" in url: tag = "[CONTACT]"
        elif "listed-companies" in url: tag = "[COMPANY_PROFILE]"
        elif "rules" in url: tag = "[REGULATION]"
        elif "news" in url: tag = "[NEWS]"
        elif "financial-results" in url: tag = "[FINANCIALS]"
        
        found_pdfs = []

        try:
            response = requests.get(url, headers=headers, timeout=25, verify=False)
            
            # PDF Handling
            if url.lower().endswith(".pdf") or 'application/pdf' in response.headers.get('Content-Type', ''):
                pdf_text = self._extract_text_from_pdf(response.content)
                if len(pdf_text) > 100:
                    # Explicitly mark PDFs as High Value for data
                    clean_text = f"{tag} SOURCE: {url}\nTYPE: OFFICIAL REPORT (PDF)\n\n" + self.clean_text_chunk(pdf_text)
                    return {"url": url, "text": clean_text}, f"📄 PDF Processed: {url}", []
                return None, f"⚠️ Empty PDF: {url}", []

            # HTML Handling
            elif response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if href.lower().endswith('.pdf'):
                        found_pdfs.append(urljoin(url, href))

                for item in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    item.decompose()
                
                text = soup.get_text(separator="\n")
                clean_text = self.clean_text_chunk(text)
                
                if len(clean_text) > 100:
                    final_text = f"{tag} SOURCE: {url}\nTYPE: Webpage\n\n{clean_text}"
                    return {"url": url, "text": final_text}, f"✅ Scraped: {url}", found_pdfs
                return None, f"⚠️ Empty: {url}", found_pdfs
            
            return None, f"❌ Failed {url}: {response.status_code}", []
        except Exception as e:
            return None, f"❌ Error {url}: {str(e)}", []

    def scrape_nse_website(self, urls):
        scraped_data = []
        logs = []
        discovered_pdfs = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(self._scrape_single_url, url): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                data, log_msg, pdfs_found = future.result()
                logs.append(log_msg)
                if data: scraped_data.append(data)
                for pdf in pdfs_found: discovered_pdfs.add(pdf)

        high_value_pdfs = [p for p in discovered_pdfs if "report" in p.lower() or "list" in p.lower() or "stat" in p.lower()]
        remaining_pdfs = list(discovered_pdfs - set(high_value_pdfs))
        final_pdf_list = (high_value_pdfs + remaining_pdfs)[:30]

        if final_pdf_list:
            logs.append(f"🔍 Parsing {len(final_pdf_list)} financial documents...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(self._scrape_single_url, url): url for url in final_pdf_list}
                for future in concurrent.futures.as_completed(future_to_url):
                    data, log_msg, _ = future.result()
                    logs.append(log_msg)
                    if data: scraped_data.append(data)

        return scraped_data, logs

    def build_knowledge_base(self, urls=None):
        try:
            self.chroma_client.delete_collection("nse_data")
        except: pass
        self.collection = self.chroma_client.get_or_create_collection(name="nse_data")

        if not urls:
            urls = [
                # Core Pages
                "https://www.nse.co.ke/",
                "https://www.nse.co.ke/site-map/",
                
                # Corporate & Leadership
                "https://www.nse.co.ke/about-nse/management-team/",
                "https://www.nse.co.ke/about-nse/board-of-directors/",
                "https://www.nse.co.ke/contact-us/",
                
                # Market Data & Stats
                "https://www.nse.co.ke/market-statistics/",
                "https://www.nse.co.ke/market-statistics/daily-market-report/",
                "https://www.nse.co.ke/dataservices/market-statistics/",
                "https://www.nse.co.ke/data-price-list/",
                
                # Listings & Products
                "https://www.nse.co.ke/listed-companies/",
                "https://www.nse.co.ke/products/equities/",
                "https://www.nse.co.ke/products/derivatives/",
                "https://www.nse.co.ke/products/reits/",
                "https://www.nse.co.ke/products/etfs/",
                "https://www.nse.co.ke/products/bonds/",
                "https://www.nse.co.ke/exchange-traded-funds/",
                "https://www.nse.co.ke/alternative-investment-market-segment/",
                
                # Investor Relations
                "https://www.nse.co.ke/investor-news/",
                "https://www.nse.co.ke/nse-investor-calendar/",
                "https://www.nse.co.ke/financial-results/",
                "https://www.nse.co.ke/news/announcements/",
                
                # Education & Rules
                "https://www.nse.co.ke/faqs/",
                "https://www.nse.co.ke/rules/",
                "https://www.cma.or.ke/regulatory-framework/"
            ]

        data, logs = self.scrape_nse_website(urls)
        if not data: return "No data found.", logs

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
        
        try:
            with open("last_update.txt", "w") as f: f.write(str(time.time()))
        except: pass

        return f"Enterprise Indexing Complete! {len(all_chunks)} chunks active.", logs

    def check_guardrails(self, query):
        """
        First line of defense: Determine if query is relevant to NSE/Finance.
        Returns (is_safe: bool, refusal_message: str)
        """
        system_prompt = """You are a Guardrail for the Nairobi Securities Exchange chatbot.
        Classify if the user query is related to:
        1. The NSE (Nairobi Securities Exchange)
        2. Financial Markets, Stocks, Bonds, Investing
        3. Kenyan Economy or Companies
        4. Trading rules or Education
        
        If YES, output 'SAFE'.
        If NO (e.g. asking about sports, politics, coding, general chat), output 'UNSAFE'.
        Only output the single word."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0,
                max_tokens=5
            )
            result = response.choices[0].message.content.strip().upper()
            if "UNSAFE" in result:
                return False, "I specialize only in the Nairobi Securities Exchange and financial markets. Please ask me a relevant question."
            return True, ""
        except:
            # Fail open if guardrail crashes (allow query)
            return True, ""

    def generate_context_queries(self, original_query):
        prompt = f"""Generate 3 specific search queries for the Nairobi Securities Exchange database to answer: "{original_query}"
        1. Exact keyword match.
        2. The broad concept/definition.
        3. The likely document type (e.g. "Management Team", "Contact Us", "Daily Report").
        Output ONLY 3 lines."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()][:3]
        except:
            return [original_query]

    def rerank_results(self, original_query, documents, sources):
        scored_results = []
        query_lower = original_query.lower()
        
        people_intent = any(w in query_lower for w in ['ceo', 'chairman', 'director', 'manager', 'who'])
        loc_intent = any(w in query_lower for w in ['location', 'address', 'where', 'contact'])
        data_intent = any(w in query_lower for w in ['price', 'value', 'volume', 'rate', 'today', 'gainers', 'losers', 'close', 'open'])
        def_intent = any(w in query_lower for w in ['what is', 'definition', 'meaning', 'explain'])
        
        for i, doc in enumerate(documents):
            score = 0
            doc_lower = doc.lower()
            source_lower = sources[i].lower()
            
            # Hybrid Search (Keyword Boost)
            # If exact words match, give massive boost
            common_words = set(query_lower.split()) & set(doc_lower.split())
            score += len(common_words) * 2 
                
            if people_intent and ("[LEADERSHIP]" in doc or "management" in source_lower):
                score += 20
            elif loc_intent and ("[CONTACT]" in doc or "contact" in source_lower):
                score += 20
            elif data_intent and ("[MARKET_DATA]" in doc or "statistics" in source_lower or "price" in source_lower):
                score += 15
            elif def_intent and ("[FAQ]" in doc or "products" in source_lower):
                score += 15
            
            if ".pdf" in source_lower and (data_intent or "report" in query_lower):
                score += 10

            scored_results.append((score, doc, sources[i]))
            
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[:7]

    def answer_question(self, query):
        try:
            # 1. Guardrails Check
            is_safe, refusal_msg = self.check_guardrails(query)
            if not is_safe:
                return refusal_msg, []

            if self.collection is None: return "System initializing...", []

            # 2. Multi-Query Retrieval
            search_queries = self.generate_context_queries(query)
            query_embeddings = self.get_embeddings_batch(search_queries)
            results = self.collection.query(query_embeddings=query_embeddings, n_results=10)
            
            raw_docs = []
            raw_sources = []
            seen = set()
            for i, doc_list in enumerate(results['documents']):
                meta_list = results['metadatas'][i]
                for j, text in enumerate(doc_list):
                    if text and text not in seen:
                        seen.add(text)
                        raw_docs.append(text)
                        raw_sources.append(meta_list[j]['source'])
            
            if not raw_docs: return "I couldn't find that information in my NSE database.", []

            # 3. Re-Ranking
            top_results = self.rerank_results(query, raw_docs, raw_sources)
            
            context_text = ""
            visible_sources = []
            for score, doc, source in top_results:
                context_text += f"\n[Source: {source}]\n{doc}\n---"
                if source not in visible_sources:
                    visible_sources.append(source)
            
            visible_sources = visible_sources[:3]

            today = datetime.date.today().strftime("%Y-%m-%d")
            
            system_prompt = f"""You are a Senior Financial Analyst for the Nairobi Securities Exchange.
            Current Date: {today}
            
            YOUR MISSION:
            Provide accurate, professional, and data-driven answers to investors.
            
            GUIDELINES:
            1. **Accuracy:** Only use the provided Context. If data is missing, admit it.
            2. **Format:** Use Markdown Tables for prices/stats. Use Bullet points for lists.
            3. **Context:** Distinguish between 'Equity Bank' (company) and 'Equities' (asset).
            4. **Citations:** Every fact must be supported by the sources provided.
            5. **Tone:** Professional and helpful.
            
            Context Data:
            {context_text}"""

            stream = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0,
                stream=True
            )
            return stream, visible_sources

        except Exception as e:
            return f"Error: {str(e)}", []