# nse_engine_v2.py
# Nairobi Securities Exchange RAG Engine – Ultimate Edition (Nov 2025)
# Implements: Hybrid Search Logic, Semantic Chunking, Live Market Status, 
# Logging, Entity Tagging, Hard Reranking Rules.

import os
import re
import json
import time
import uuid
import random
import datetime
import hashlib
import logging
import requests
import urllib3
import chromadb
import pdfplumber
import io
from typing import List, Tuple, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

# Suppress warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------- CONFIGURATION -------------------
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
MAX_CRAWL_DEPTH = 2
MAX_PAGES_TO_CRAWL = 800 
CHROMA_PATH = "./nse_db_v2"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nse_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NSE-Engine")

class NSEKnowledgeBase:
    def __init__(self, openai_api_key: str):
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required")
        
        self.api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.session = requests.Session()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            self.collection = self.chroma_client.get_or_create_collection(name="nse_data_v2")
        except Exception as e:
            logger.error(f"Failed to init ChromaDB: {e}")
            self.collection = None

    # --- HELPER: LIVE MARKET STATUS ---
    def get_market_status(self) -> str:
        """Returns the current trading status of the NSE."""
        now = datetime.datetime.now() + datetime.timedelta(hours=3) # Approx EAT if server is UTC
        # Adjust for Kenya Time (UTC+3) if needed, assuming server is UTC
        
        if now.weekday() >= 5:
            return "[MARKET CLOSED – Weekend]"
        
        hour = now.hour
        if hour < 9:
            return f"[MARKET CLOSED – Opens at 09:00 | Current: {now.strftime('%H:%M')}]"
        elif hour < 9 or (hour == 9 and now.minute < 30):
            return f"[PRE-OPEN SESSION | {now.strftime('%H:%M')}]"
        elif 9 <= hour < 15:
            return f"[MARKET OPEN – LIVE TRADING | {now.strftime('%H:%M')}]"
        else:
            return f"[MARKET CLOSED – Closed at 15:00 | {now.strftime('%H:%M')}]"

    # --- HELPER: STATIC KNOWLEDGE ---
    def get_static_facts(self):
        return """
        [OFFICIAL_FACT_SHEET]
        TOPIC: NSE Leadership, Structure & Market Rules
        SOURCE: NSE Official Website / Annual Report 2025
        LAST_VERIFIED: November 2025

        CEO: Mr. Frank Mwiti
        Chairman: Mr. Kiprono Kittony
        Trading Hours: Mon-Fri, 09:30–15:00 (Continuous)
        Currency: KES
        Regulator: Capital Markets Authority (CMA)
        Depository: CDSC
        Key Indices: NASI (All Share), NSE 20, NSE 25
        Settlement: T+3 (Equities), T+1/T+3 (Bonds)
        """

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
    
    def is_data_stale(self):
        last = self.get_last_update_time()
        return (time.time() - last) > 86400 # 24 hours

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_embeddings_batch(self, texts):
        if not texts: return []
        # Batch optimization
        if len(texts) > 100:
            results = []
            for i in range(0, len(texts), 100):
                batch = texts[i:i+100]
                results.extend(self.get_embeddings_batch(batch))
                time.sleep(0.2)
            return results

        sanitized_texts = [t.replace("\n", " ") for t in texts]
        try:
            response = self.client.embeddings.create(input=sanitized_texts, model=EMBEDDING_MODEL)
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Embedding Error: {e}")
            raise e

    def get_embedding(self, text):
        return self.get_embeddings_batch([text])[0]

    # --- SEMANTIC CHUNKING ---
    def clean_text_chunk(self, text):
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def simple_text_splitter(self, text, chunk_size=1200, overlap=300):
        """
        Smarter splitting that respects sentence boundaries.
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            
            # Try to find a sentence ending (.!?) to break cleanly
            if end < text_len:
                lookahead = text[start:end+50] # Peek ahead slightly
                last_period = -1
                for p in ['. ', '? ', '! ', '\n']:
                    idx = text.rfind(p, start, end)
                    if idx > last_period:
                        last_period = idx
                
                if last_period != -1 and last_period > start + (chunk_size // 2):
                    end = last_period + 1
            
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
            if start >= end: start = end # Prevent infinite loops
            
        return chunks

    # --- STRUCTURED PDF EXTRACTION ---
    def _extract_text_from_pdf(self, pdf_content):
        """Uses pdfplumber for table-aware extraction."""
        text_parts = []
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for page in pdf.pages:
                    # 1. Extract tables specifically
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            # Convert to Markdown Table
                            headers = table[0]
                            rows = table[1:]
                            # Filter out None values
                            safe_headers = [str(h).replace('\n', ' ') if h else "" for h in headers]
                            md_table = "| " + " | ".join(safe_headers) + " |\n"
                            md_table += "| " + " --- |" * len(safe_headers) + "\n"
                            for row in rows:
                                safe_row = [str(c).replace('\n', ' ') if c else "" for c in row]
                                md_table += "| " + " | ".join(safe_row) + " |\n"
                            text_parts.append(f"[TABLE]\n{md_table}")

                    # 2. Extract regular text
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"PDF Extraction failed: {e}")
            return ""

    # --- CRAWLING ---
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_url(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Upgrade-Insecure-Requests': '1',
        }
        return requests.get(url, headers=headers, timeout=20, verify=False)

    def crawl_site(self, seed_urls):
        visited = set()
        to_visit = set(seed_urls)
        found_content = set()
        found_pdfs = set()
        
        # Explicitly add the hardcoded PDF list you provided
        hardcoded_pdfs = [
             "https://www.nse.co.ke/wp-content/uploads/Safaricom-PLC-Announcement-of-an-Interim-Dividend-For-The-Year-Ended-31-03-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Kenya-Orchards-Ltd-Cautionary-Announcement.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Equity-Group-Holdings-Plc-EQUITY-GROUP-HOLDINGS-PLC-CHANGE-OF-BOARD.pdf",
            "https://www.nse.co.ke/wp-content/uploads/StanChart-Corporate-Calendar-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/TotalEnergies-Marketing-Kenya-PLC-Corporate-Events-Calendar-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Derivatives_Pricelist_23-FEB-2023.pdf",
            "https://www.nse.co.ke/wp-content/uploads/27-NOV-23.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-equities-trading-rules.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-derivatives-rules.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-listing-rules.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-market-participants-rules.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Equity-Trading-Rules-Amended-Jul-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/policy-guidance-note-for-green-bonds.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-OPERATIONAL-GUIDELINES-FOR-THE-BOND-QUOTATIONS-BOARD.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-fixed-income-trading-rules.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Equity-Group-Holdings-Plc-Unaudited-Financial-Statements-Other-Disclosures-for-the-Period-Ended-30-Sep-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/The-Kenya-Power-Lighting-Company-Plc-Audited-Financial-Results-for-the-Year-Ended-30-Jun-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Limuru-Tea-Plc-Unaudited-Results-for-the-Six-Months-Ended-30-06-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE_Press-Release-Nairobi-Securities-Exchange-Plc-takes-a-bold-step-to-expand-investment-access-for-retail-investors-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Press-Release-Nairobi-Securities-Exchange-Plc-Admits-Fintrust-Securities-Limited-as-an-Authorized-Securities-Dealer-ASD-in-the-Fixed-Income-Market_-update_-16_0.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Investors-on-Kenyas-securities-exchange-will-soon-get-access-to-trade-global-markets-as-Satrix-lists-MSCI-World-Feeder-ETF-on-the-NSE.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Press-Release-Nairobi-Securities-Exchange-Plc-Appoints-Sterling-Capital-Limited-as-a-market-maker-in-the-NEXT-Derivatives-Market.pdf",
            "https://www.nse.co.ke/wp-content/uploads/board-diversity-inclusion-2021-kim-research-report.pdf",
            "https://www.nse.co.ke/wp-content/uploads/equileap_kenya-report-2019_final_print.pdf",
            "https://www.nse.co.ke/wp-content/uploads/FOB_Book_Digital_2020.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Groundrules-nse-NSE-BankingSector-share-index_-V1.0.pdf",
            "https://www.nse.co.ke/wp-content/uploads/GroundRules-NSE-Bond-Index-NSE-BI-v2-Index-final-2.pdf",
            "https://www.nse.co.ke/wp-content/uploads/groundrules-nse-20-share-index_-v1.6.pdf",
            "https://www.nse.co.ke/wp-content/uploads/GroundRules-NSE-10v2-Share-Index.pdf",
            "https://www.nse.co.ke/wp-content/uploads/groundrules-nse-25-share-index_-v1.4.pdf",
            "https://www.nse.co.ke/wp-content/uploads/East-African-Exchanges-EAE-20-Share-Index-Methodology-F-.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Fixed-Income-Trading-Rules-2024.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Implied-Yields-Yield-Curve-Generation-Methodology-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-ESG-Disclosures-Guidance-Manual.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Day-Trading-Operational-Guidelines.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-direct-market-access-october-2019.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-financial-resource-requirements-for-market-intermediaries.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-managementsupervision-and-internal-control-of-cma-licensed-entities-may-2012.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-the-prevention-of-money-laundering-and-terrorism-financing-in-the-capital-markets.pdf",
             "https://www.nse.co.ke/wp-content/uploads/Carbacid-Investments-Plc-Audited-Group-Results-for-the-Year-Ended-31st-July-2025-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Olympia-Capital-Holdings-Limited-Half-year-unaudited-reports-as-at-August-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/KenGen-Audited-Results-for-the-Year-Ended-30th-June-2025-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/KenGen-Audited-Results-for-the-Year-Ended-30th-June-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-ESOP-Public-Announcement-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Nairobi-Securities-Exchange-Plc-Appointment-of-Non-Executive-Director.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Press-Release-Nairobi-Securities-Exchange-Plc-Launches-Banking-Sector-Index.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Nairobi-Securities-Exchange-Plc-Notice-of-Appointment-of-Director.pdf",
            "https://www.nse.co.ke/wp-content/uploads/03-NOV-25.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Derivatives_Pricelist_24-OCT-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/boarddiversityreport.pdf",
            "https://www.nse.co.ke/wp-content/uploads/RENAISSANCE-CAPITAL.pdf",
            "https://www.nse.co.ke/wp-content/uploads/STERLING-CAPITAL.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Derivatives_Pricelist_28-APR-2023.pdf",
            "https://www.nse.co.ke/wp-content/uploads/05-NOV-24.pdf",
            "https://www.nse.co.ke/wp-content/uploads/05-FEB-24.pdf",
            "https://www.nse.co.ke/wp-content/uploads/BondPrices_25-APR-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/BondPrices_29-SEP-2023.pdf",
            "https://www.nse.co.ke/wp-content/uploads/BondPrices_28-OCT-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/KCB-Capital.pdf",
            "https://www.nse.co.ke/wp-content/uploads/BondPrices_15-JUN-2022.pdf",
            "https://www.nse.co.ke/wp-content/uploads/68th-Annual-General-Meeting-Shareholder-Questions-Responses.pdf",
            "https://www.nse.co.ke/wp-content/uploads/68th-NSE-AGM-Presentation.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-67th-annual-general-meeting-chief-executive-presentation.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE66AnnualGeneralMeetingCEPresentation.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-press-briefing-presentation-final.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-2018-half-year-results-.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-2017-half-year-results-presentation-22-.08.2017.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-2015-full-year-results-ceo-presentation_version-3_24-03-2016.pdf",
            
            # Newly added PDFs
            "https://www.nse.co.ke/wp-content/uploads/East-Africa-Debt-Capital-Markets-Masterclass-brochure.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Masterclass-brochure-final-1-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-training-calendar-2024-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/ESG-training-brochure-August-2024.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Sharia-updated-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/KenGen-Audited-Results-for-the-Year-Ended-30th-June-2025-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-ESOP-Public-Announcement-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Carbacid-Investments-Plc-Audited-Group-Results-for-the-Year-Ended-31st-July-2025-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Olympia-Capital-Holdings-Limited-Half-year-unaudited-reports-as-at-August-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Nairobi-Securities-Exchange-Plc-Notice-of-Appointment-of-Director.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Nairobi-Securities-Exchange-Plc-Appointment-of-Non-Executive-Director.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Press-Release-Nairobi-Securities-Exchange-Plc-Launches-Banking-Sector-Index.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Market-Data-Pricelist.pdf",
            "https://www.nse.co.ke/wp-content/uploads/broker-back-office-prequalified-vendors.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-2025-2029-Strategy.pdf",
            "https://www.nse.co.ke/wp-content/uploads/BBO-standards.pdf"
        ]
        
        count = 0
        while to_visit and count < MAX_PAGES_TO_CRAWL:
            url = to_visit.pop()
            if url in visited: continue
            visited.add(url)
            
            if "nse.co.ke" not in url: continue
            
            try:
                if url.lower().endswith(".pdf"):
                    found_pdf_urls.add(url)
                    continue

                response = self._fetch_url(url)
                if response.status_code != 200: continue
                
                if 'application/pdf' in response.headers.get('Content-Type', ''):
                    found_pdf_urls.add(url)
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                found_content.add(url)
                count += 1
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    if "nse.co.ke" in full_url:
                        if full_url.lower().endswith(".pdf"):
                            found_pdf_urls.add(full_url)
                        elif full_url not in visited and full_url not in to_visit:
                            if len(to_visit) < 100: # Limited queue
                                to_visit.add(full_url)
            except:
                pass
        
        all_pdfs = list(found_pdf_urls.union(set(hardcoded_pdfs)))
        return list(found_content), all_pdfs

    def _process_content(self, url, content_type, content_bytes):
        text = ""
        tag = "[GENERAL]"
        # Auto-tagging
        if "statistics" in url: tag = "[MARKET_DATA]"
        elif "management" in url or "directors" in url: tag = "[LEADERSHIP]"
        elif "contact" in url: tag = "[CONTACT]"
        elif "rules" in url: tag = "[REGULATION]"
        elif "news" in url: tag = "[NEWS]"
        elif "financial" in url or "result" in url: tag = "[FINANCIALS]"
        elif "calendar" in url: tag = "[CALENDAR]"
        elif "strategy" in url: tag = "[STRATEGY]"
        elif "guidelines" in url: tag = "[GUIDELINES]"
        elif "corporate-actions" in url: tag = "[CORPORATE_ACTION]"
        elif "circulars" in url: tag = "[CIRCULAR]"
        elif "listed-companies" in url: tag = "[COMPANY_DATA]"
        elif "derivatives" in url: tag = "[DERIVATIVES]"
        elif "csr" in url: tag = "[CSR]"
        elif "e-digest" in url: tag = "[DIGEST]"
        elif "press-releases" in url: tag = "[PRESS_RELEASE]"
        elif "publications" in url: tag = "[PUBLICATION]"
        elif "trading" in url: tag = "[TRADING]"

        if content_type == "pdf":
            raw_text = self._extract_text_from_pdf(content_bytes)
            if len(raw_text) > 100:
                text = f"{tag} SOURCE: {url}\nTYPE: OFFICIAL REPORT (PDF)\n\n" + self.clean_text_chunk(raw_text)
        else:
            soup = BeautifulSoup(content_bytes, 'html.parser')
            for item in soup(["script", "style", "nav", "footer", "header"]):
                item.decompose()
            raw_text = soup.get_text(separator="\n")
            clean = self.clean_text_chunk(raw_text)
            if len(clean) > 100:
                text = f"{tag} SOURCE: {url}\nTYPE: Webpage\n\n{clean}"
        return text

    def scrape_and_index(self, urls, content_type="html"):
        """Scrapes list of URLs and indexes them ONLY if changed (Saves Credits)"""
        
        new_chunks = 0
        
        def task(url):
            try:
                response = self._fetch_url(url)
                if response.status_code == 200: return url, response.content
            except: pass
            return None, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(task, url): url for url in urls}
            
            for future in concurrent.futures.as_completed(futures):
                url, content = future.result()
                if content:
                    text = self._process_content(url, content_type, content)
                    if not text: continue

                    # --- EFFICIENCY UPGRADE: HASH CHECK ---
                    current_hash = self.compute_hash(text)
                    
                    # Check DB for existing version of this URL
                    existing = self.collection.get(
                        where={"source": url},
                        include=["metadatas"]
                    )
                    
                    # If we already have this page indexed
                    if existing['ids']:
                        # Check if the content hash matches
                        # We grab the hash from the first chunk of this URL
                        stored_hash = existing['metadatas'][0].get("page_hash", "")
                        
                        if stored_hash == current_hash:
                            # Content hasn't changed -> SKIP EMBEDDING (Saves $$$)
                            continue
                        else:
                            # Content changed -> Delete old chunks to avoid duplicates
                            self.collection.delete(where={"source": url})
                    # --------------------------------------
                    
                    chunks = self.simple_text_splitter(text)
                    ids = [str(uuid.uuid4()) for _ in chunks]
                    
                    # Store the hash in metadata so we can check it next time
                    metadatas = [{"source": url, "date": datetime.date.today().isoformat(), "page_hash": current_hash} for _ in chunks]
                    
                    embeddings = self.get_embeddings_batch(chunks)
                    if embeddings:
                        self.collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
                        new_chunks += len(chunks)
        return new_chunks

    def build_knowledge_base(self):
        # 1. Comprehensive Seed List
        seeds = [
            "https://www.nse.co.ke/",
            "https://www.nse.co.ke/home/",
            "https://www.nse.co.ke/about-nse/",
            "https://www.nse.co.ke/about-nse/history/",
            "https://www.nse.co.ke/about-nse/vision-mission/",
            "https://www.nse.co.ke/about-nse/board-of-directors/",
            "https://www.nse.co.ke/about-nse/management-team/",
            "https://www.nse.co.ke/leadership/",
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
            "https://www.nse.co.ke/indices/nse-bond-index/",
            "https://academy.nse.co.ke/",
            "https://www.nse.co.ke/nominated-advisors/",
            "https://www.nse.co.ke/corporate-actions/",
            "https://www.nse.co.ke/investor-presentations/",
            "https://onlinetrading.nse.co.ke/",
            "https://www.nse.co.ke/listed-company-announcements/",
            "https://www.nse.co.ke/financial-results/",
            "https://www.nse.co.ke/press-releases/",
            "https://www.nse.co.ke/publications/",
            "https://www.nse.co.ke/rules/",
            "https://www.nse.co.ke/guidelines/",
            "https://www.nse.co.ke/site-map/",
            "https://www.nse.co.ke/mobile-and-online-trading/",
            "https://www.nse.co.ke/dataservices/historical-data-request-form/",
            "https://www.nse.co.ke/derivatives/about-next"
            "https://www.nse.co.ke/investor-news/",
            
            # Newly added links
            "https://www.nse.co.ke/growth-enterprise-market-segment/",
            "https://www.nse.co.ke/real-estate-investment-trusts/",
            "https://www.nse.co.ke/main-investment-market-segment/",
            "https://www.nse.co.ke/equities-market/",
            "https://www.nse.co.ke/exchange-traded-funds/",
            "https://www.nse.co.ke/green-bonds/",
            "https://www.nse.co.ke/alternative-investment-market-segment/",
            "https://www.nse.co.ke/training/",
            "https://www.nse.co.ke/careers/",
            "https://www.nse.co.ke/leadership/",
            "https://www.nse.co.ke/about-nse/",
            "https://www.nse.co.ke/our-story/",
            "https://www.nse.co.ke/cookies-policy/",
            "https://www.nse.co.ke/contact-us/",
            "https://www.nse.co.ke/privacy-policy/",
            "https://www.nse.co.ke/tenders/",
            "https://www.nse.co.ke/mobile-and-online-trading/",
            "https://www.nse.co.ke/listed-company-announcements/",
            "https://www.nse.co.ke/corporate-bonds/",
            "https://www.nse.co.ke/faqs/",
            "https://www.nse.co.ke/the-east-africa-islamic-finance-forum-2025/",
            "https://www.nse.co.ke/government-bonds/",
            "https://www.nse.co.ke/ibuka-2/",
            "https://www.nse.co.ke/m-akiba/",
            "https://www.nse.co.ke/usp/",
            
            # New Data Services Links
            "https://www.nse.co.ke/dataservices/market-statistics/",
            "https://www.nse.co.ke/dataservices/real-time-data/",
            "https://www.nse.co.ke/dataservices/end-of-day-data/",
            "https://www.nse.co.ke/dataservices/historical-data/",
            "https://www.nse.co.ke/dataservices/international-securities-identification-number-isin/",
            "https://www.nse.co.ke/dataservices/api-specification-documents/",
            
            # Additional links added
            "https://www.nse.co.ke/dataservices/",
            "https://www.nse.co.ke/dataservices/market-data-overview/",
            "https://www.nse.co.ke/sustainability/",
            "https://www.nse.co.ke/login/",
            "https://www.nse.co.ke/cart/",
            "https://www.nse.co.ke/nominated-advisors/",
            
            # Latest Batch
            "https://www.nse.co.ke/guidelines/",
            "https://www.nse.co.ke/corporate-actions/",
            "https://www.nse.co.ke/rules/",
            "https://www.nse.co.ke/listed-companies/",
            "https://www.nse.co.ke/listed-company-announcements/",
            "https://www.nse.co.ke/derivatives/",
            "https://www.nse.co.ke/nse-investor-calendar/",
            "https://www.nse.co.ke/policy-guidance-notes/",
            "https://www.nse.co.ke/circulars/",
            
            # More Latest Batch
            "https://www.nse.co.ke/how-to-become-a-trading-participant/",
            "https://www.nse.co.ke/e-digest/",
            "https://www.nse.co.ke/list-of-trading-participants/",
            "https://www.nse.co.ke/nse-events/",
            "https://www.nse.co.ke/csr/",
            "https://www.nse.co.ke/press-releases/",
            "https://www.nse.co.ke/publications/",
            "https://www.nse.co.ke/trading-participant-financials/"
        ]
        
        print("🕷️ Crawling NSE website...")
        discovered_pages, discovered_pdfs = self.crawl_site(seeds)
        all_pages = list(set(discovered_pages))
        
        print(f"📝 Found {len(all_pages)} pages and {len(discovered_pdfs)} PDFs.")
        
        # Reset DB
        try: self.chroma_client.delete_collection("nse_data_v2")
        except: pass
        self.collection = self.chroma_client.get_or_create_collection(name="nse_data_v2")
        
        chunks_1 = self.scrape_and_index(all_pages, "html")
        chunks_2 = self.scrape_and_index(discovered_pdfs, "pdf") 
        
        try:
            with open("last_update.txt", "w") as f: f.write(str(time.time()))
        except: pass
        
        return f"Knowledge Base Updated: {chunks_1 + chunks_2} chunks indexed.", []

    def generate_context_queries(self, original_query):
        today = datetime.date.today().strftime("%Y-%m-%d")
        prompt = f"""Generate 3 search queries for the NSE database for: "{original_query}"
        Current Date: {today}
        Output ONLY 3 lines."""
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3
            )
            return [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()][:3]
        except:
            return [original_query]

    def llm_rerank(self, query, documents, sources):
        if not documents: return []
        candidates = ""
        for i, doc in enumerate(documents):
            candidates += f"\n--- DOC {i} ---\nSource: {sources[i]}\nContent: {doc[:300]}...\n"
        
        prompt = f"""Rank these documents by relevance to: "{query}".
        Return IDs of top 5 documents (e.g. 0, 3, 1).
        Documents: {candidates}"""
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
            )
            indices = [int(x) for x in re.findall(r'\d+', response.choices[0].message.content)]
            reranked = []
            for idx in indices:
                if idx < len(documents): reranked.append((documents[idx], sources[idx]))
            return reranked
        except:
            return list(zip(documents, sources))

    def answer_question(self, query):
        try:
            if self.collection is None: return "System initializing...", []
            search_queries = self.generate_context_queries(query)
            query_embeddings = self.get_embeddings_batch(search_queries)
            results = self.collection.query(query_embeddings=query_embeddings, n_results=15)
            
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

            top_results = self.llm_rerank(query, raw_docs, raw_sources)
            context_text = ""
            visible_sources = []
            for doc, source in top_results[:5]: 
                context_text += f"\n[Source: {source}]\n{doc}\n---"
                if source not in visible_sources: visible_sources.append(source)

            today = datetime.date.today().strftime("%Y-%m-%d")
            system_prompt = f"""You are the NSE Digital Assistant.
            TODAY'S DATE: {today}
            MARKET STATUS: {self.get_market_status()}
            
            INSTRUCTIONS:
            1. **Accuracy:** Use CONTEXT only.
            2. **Date Awareness:** Prioritize 2024/2025 data.
            3. **Formatting:** Use Markdown.
            
            CONTEXT:
            {context_text}"""

            stream = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                temperature=0,
                stream=True
            )
            return stream, visible_sources
        except Exception as e:
            return f"Error: {str(e)}", []