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
import random
import datetime
import hashlib
from pypdf import PdfReader
from urllib.parse import urljoin, urlparse
from tenacity import retry, stop_after_attempt, wait_fixed

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
MAX_CRAWL_DEPTH = 2
MAX_PAGES_TO_CRAWL = 50

class NSEKnowledgeBase:
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required")
        
        self.api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        
        self.db_path = "./nse_db_pure"
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Persistent session for faster crawling
        self.session = requests.Session()
        
        try:
            self.collection = self.chroma_client.get_or_create_collection(name="nse_data")
        except Exception:
            self.collection = None

    # --- STATIC KNOWLEDGE (THE CHEAT SHEET) ---
    def get_static_facts(self):
        """
        Returns high-value, static facts that are often hard to crawl.
        This acts as a 'Knowledge Anchor' for the bot.
        """
        return """
        [OFFICIAL_FACT_SHEET]
        TOPIC: NSE Leadership & Key Facts
        SOURCE: Manual Verification / NSE Official Documents
        LAST_VERIFIED: 2025
        
        BOARD OF DIRECTORS (NSE PLC):
        1. Mr. Kiprono Kittony - Chairman
        2. Mr. Paul Mwai - Vice-Chairman
        3. Mr. Frank Mwiti - Chief Executive Officer (Appointed May 2, 2024)
        4. Ms. Risper Alaro-Mukoto - Non-Executive Director
        5. Mr. Stephen Chege - Non-Executive Director
        6. Mrs. Isis Madison - Independent Non-Executive Director
        7. Mr. John Niepold - Independent Non-Executive Director
        8. Mr. Donald Wangunyu - Non-Executive Director
        9. Mrs. Caroline Kariuki - Independent Non-Executive Director
        
        MANAGEMENT TEAM:
        1. Frank Mwiti - CEO
        2. David Wainaina - Chief Operating Officer
        3. Jane Kiarie - Chief Financial Officer
        
        CONTACT & OPERATIONS:
        - Address: 55 Westlands Road, Nairobi, Kenya
        - Trading Hours: 09:30 AM - 03:00 PM (Monday - Friday)
        - Regulator: Capital Markets Authority (CMA)
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
        """Checks if data is older than 24 hours"""
        last_update = self.get_last_update_time()
        if last_update == 0: return True
        return (time.time() - last_update) > 86400

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def get_embeddings_batch(self, texts):
        if not texts: return []
        if len(texts) > 100:
            results = []
            for i in range(0, len(texts), 100):
                batch = texts[i:i+100]
                results.extend(self.get_embeddings_batch(batch))
                time.sleep(0.5)
            return results

        sanitized_texts = [t.replace("\n", " ") for t in texts]
        try:
            response = self.client.embeddings.create(input=sanitized_texts, model=EMBEDDING_MODEL)
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Embedding Error: {e}")
            raise e

    def clean_text_chunk(self, text):
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def simple_text_splitter(self, text, chunk_size=1500, overlap=300):
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + (chunk_size // 2):
                    end = last_period + 1
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
            if start >= end: start = end
        return chunks

    def _extract_text_from_pdf(self, pdf_content):
        try:
            pdf_file = io.BytesIO(pdf_content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                try:
                    page_text = page.extract_text(extraction_mode="layout")
                except:
                    page_text = page.extract_text()
                text += page_text + "\n"
            return text
        except Exception:
            return ""

    def _get_random_header(self):
        """Rotates User-Agents to avoid detection"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ]
        return {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_url(self, url):
        # Random sleep to be polite and avoid rate limiting
        time.sleep(random.uniform(0.5, 1.5)) 
        return self.session.get(url, headers=self._get_random_header(), timeout=25, verify=False)

    def crawl_site(self, seed_urls):
        """
        Improved recursive crawler with better headers and targeted PDF list.
        """
        visited = set()
        to_visit = set(seed_urls)
        found_content_urls = set() 
        found_pdf_urls = set()     
        
        # UPDATED HARDCODED LIST - Includes Annual Reports & Governance Docs
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
                found_content_urls.add(url)
                count += 1
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    
                    if "nse.co.ke" in full_url:
                        if full_url.lower().endswith(".pdf"):
                            found_pdf_urls.add(full_url)
                        elif full_url not in visited and full_url not in to_visit:
                            if len(to_visit) < 100: # Increased queue size slightly
                                to_visit.add(full_url)
                                
            except Exception as e:
                print(f"Crawl Error {url}: {e}")
        
        all_pdfs = list(found_pdf_urls.union(set(hardcoded_pdfs)))
        return list(found_content_urls), all_pdfs

    def _process_content(self, url, content_type, content_bytes):
        text = ""
        tag = "[GENERAL]"
        
        # Simple Tagging System
        if "statistics" in url: tag = "[MARKET_DATA]"
        elif "leadership" in url or "board" in url: tag = "[LEADERSHIP]"
        elif "rules" in url: tag = "[REGULATION]"
        elif "financial" in url: tag = "[FINANCIALS]"
        elif "etf" in url or "bond" in url: tag = "[PRODUCT]"

        if content_type == "pdf":
            raw_text = self._extract_text_from_pdf(content_bytes)
            if len(raw_text) > 100:
                text = f"{tag} SOURCE: {url}\nTYPE: OFFICIAL PDF\n\n" + self.clean_text_chunk(raw_text)
        else:
            soup = BeautifulSoup(content_bytes, 'html.parser')
            for item in soup(["script", "style", "nav", "footer", "header", "aside"]):
                item.decompose()
            raw_text = soup.get_text(separator="\n")
            clean = self.clean_text_chunk(raw_text)
            if len(clean) > 100:
                text = f"{tag} SOURCE: {url}\nTYPE: Webpage\n\n{clean}"
        
        return text

    def scrape_and_index(self, urls, content_type="html"):
        new_chunks = 0
        
        def task(url):
            try:
                response = self._fetch_url(url)
                if response.status_code == 200:
                    return url, response.content
            except:
                pass
            return None, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor: # Reduced workers to be nicer to server
            futures = {executor.submit(task, url): url for url in urls}
            
            for future in concurrent.futures.as_completed(futures):
                url, content = future.result()
                if content:
                    text = self._process_content(url, content_type, content)
                    if not text: continue

                    chunks = self.simple_text_splitter(text)
                    ids = [str(uuid.uuid4()) for _ in chunks]
                    metadatas = [{"source": url, "date": datetime.date.today().isoformat()} for _ in chunks]
                    
                    embeddings = self.get_embeddings_batch(chunks)
                    
                    if embeddings:
                        self.collection.add(
                            documents=chunks,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            ids=ids
                        )
                        new_chunks += len(chunks)
        return new_chunks

    def build_knowledge_base(self):
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
        
        print("🕷️ Crawling NSE website with enhanced headers...")
        discovered_pages, discovered_pdfs = self.crawl_site(seeds)
        
        try:
            self.chroma_client.delete_collection("nse_data")
        except: pass
        self.collection = self.chroma_client.get_or_create_collection(name="nse_data")
        
        chunks_1 = self.scrape_and_index(list(set(discovered_pages)), "html")
        chunks_2 = self.scrape_and_index(discovered_pdfs, "pdf") 
        
        try:
            with open("last_update.txt", "w") as f: f.write(str(time.time()))
        except: pass
        
        return f"Updated: {chunks_1 + chunks_2} chunks indexed.", []

    def generate_context_queries(self, original_query):
        today = datetime.date.today().strftime("%Y-%m-%d")
        prompt = f"""Generate 3 search queries for the NSE database based on: "{original_query}"
        
        RULES:
        1. If the user asks for "Board" or "CEO", specifically search for "NSE Board of Directors" and "Leadership".
        2. Ensure all queries imply the Nairobi Securities Exchange context.
        3. Output ONLY the 3 queries, one per line.
        """
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3
            )
            return [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()][:3]
        except:
            return [original_query + " NSE", "Nairobi Securities Exchange " + original_query]

    def llm_rerank(self, query, documents, sources):
        if not documents: return []
        
        candidates = ""
        for i, doc in enumerate(documents):
            candidates += f"\n--- DOC {i} ---\nSource: {sources[i]}\nContent: {doc[:300]}...\n"
            
        prompt = f"""Rank these documents by relevance to: "{query}" (Context: NSE Kenya).
        Return IDs of top 5 (e.g., "0, 3, 1").
        Prioritize [LEADERSHIP] or [OFFICIAL PDF] tags if the user asks about rules or people.
        
        Documents:
        {candidates}"""
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            indices = [int(x) for x in re.findall(r'\d+', response.choices[0].message.content)]
            
            reranked = []
            for idx in indices:
                if idx < len(documents):
                    reranked.append((documents[idx], sources[idx]))
            return reranked
        except:
            return list(zip(documents, sources))

    def answer_question(self, query):
        try:
            if self.is_data_stale(): pass 

            if self.collection is None: return "System initializing...", []

            # 1. Generate Search Queries
            search_queries = self.generate_context_queries(query)
            
            # 2. Retrieve
            query_embeddings = self.get_embeddings_batch(search_queries)
            results = self.collection.query(query_embeddings=query_embeddings, n_results=15)
            
            raw_docs = []
            raw_sources = []
            seen = set()
            
            if results['documents']:
                for i, doc_list in enumerate(results['documents']):
                    meta_list = results['metadatas'][i]
                    for j, text in enumerate(doc_list):
                        if text and text not in seen:
                            seen.add(text)
                            raw_docs.append(text)
                            raw_sources.append(meta_list[j]['source'])
            
            # 3. Re-rank
            top_results = self.llm_rerank(query, raw_docs, raw_sources)
            
            # 4. INJECT STATIC FACTS (The "Cheat Sheet")
            # This ensures the bot knows the Board/CEO even if retrieval fails
            context_text = self.get_static_facts() + "\n\n--- RETRIEVED DATA ---\n"
            
            visible_sources = []
            for doc, source in top_results[:5]: 
                context_text += f"\n[Source: {source}]\n{doc}\n---"
                if source not in visible_sources: visible_sources.append(source)

            # 5. Final Response
            today = datetime.date.today().strftime("%Y-%m-%d")
            
            system_prompt = f"""You are the NSE Digital Assistant.
            
            CRITICAL CONTEXT RULES:
            1. **Scope:** All questions are about the Nairobi Securities Exchange (NSE) unless explicitly stated otherwise.
            2. **Ambiguity Handler:** If the user asks "Who are the directors?" without naming a company, assume they mean the **NSE's own Board of Directors** (provided in the context), BUT add a note: "If you meant the directors of a specific listed company (like Safaricom or KCB), please specify the company name."
            3. **Static Facts:** Trust the [OFFICIAL_FACT_SHEET] at the top of the context for Leadership, Addresses, and Trading Hours.
            4. **Sources:** Base your answer ONLY on the provided Context.
            
            TODAY'S DATE: {today}
            
            CONTEXT:
            {context_text}"""

            stream = self.client.chat.completions.create(
                model=LLM_MODEL,
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