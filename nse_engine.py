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
# OPTIMIZED FOR DEEPER SEARCH AS DISCUSSED
MAX_CRAWL_DEPTH = 4
MAX_PAGES_TO_CRAWL = 300

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
        return """
         [OFFICIAL_FACT_SHEET]
        TOPIC: NSE Leadership, Structure & Market Rules
        SOURCE: Manual Verification / NSE Official Documents
        LAST_VERIFIED: 2025
        
        1. LEADERSHIP (NSE PLC)
        - Chairman: Mr. Kiprono Kittony
        - Vice-Chairman: Mr. Paul Mwai
        - CEO: Mr. Frank Mwiti (Appointed May 2, 2024)
        - Board Members: Ms. Risper Alaro-Mukoto, Mr. Stephen Chege, Mrs. Isis Madison, Mr. John Niepold, Mr. Donald Wangunyu, Mrs. Caroline Kariuki.
        - Management: David Wainaina (COO), Jane Kiarie (CFO).

        2. OPERATIONAL DETAILS
        - Address: 55 Westlands Road, Nairobi, Kenya.
        - Website: www.nse.co.ke
        - Currency: Kenyan Shilling (KES / Ksh).
        - Regulator: Capital Markets Authority (CMA).
        - Depository: CDSC (Central Depository & Settlement Corporation) - Handles share holding accounts (CDS Accounts).

        3. TRADING HOURS (Monday - Friday, Excluding Public Holidays)
        - Pre-Open Session: 09:00 AM - 09:30 AM
        - Continuous Trading: 09:30 AM - 03:00 PM
        - Closing Session: 03:00 PM onwards

        4. MARKET SEGMENTS
        - MIMS (Main Investment Market Segment): For large, established companies.
        - AIMS (Alternative Investment Market Segment): For mid-sized companies.
        - GEMS (Growth Enterprise Market Segment): For SMEs and growth companies.
        - FIMS (Fixed Income Market Segment): For Corporate and Government Bonds.
        - NEXT (Derivatives Market): For Futures and Options trading.

        5. INDICES DEFINITIONS
        - NASI (NSE All Share Index): Tracks performance of ALL listed companies. Best for overall market health.
        - NSE 20 Share Index: Tracks the top 20 "Blue Chip" companies. Best for tracking stability.
        - NSE 25 Share Index: Tracks top 25 companies based on liquidity and market capitalization.

        6. SETTLEMENT CYCLES
        - Equities (Shares): T+3 (Transaction date + 3 business days).
        - Bonds: T+1 (typically) or T+3 depending on the bond type.

        7. OFFICIAL FAQs (Verbatim Answers)
        Q:What products are traded at the Nairobi Securities Exchange?
        A:The products traded at the NSE are Shares and Bonds. Shares and Bonds are money or financial products. Another name for Shares is Equities, while Bonds are also known as Debt Instruments.

        Q:What is different the difference between a stock market and any other market?
        A:The main difference about the Stock Exchange market from other local markets is in the types of products traded, how they are traded and how they are paid for and transferred.

        Q:What is the NSE?
        A:The Nairobi Securities Exchange is a Market, and commonly known as the NSE.

        Q:What is the importance of this market to the economy?
        A:For an economy to grow, money needs to shift from les to more productive activities. In other words, idle money and savings should be invested in productive activity for the economy to grow. The Nairobi Stock Exchange makes this possible by:

Enabling idle money and savings to become productive by bringing the borrowers and lenders of money together at a low cost. The lenders (all savers) become the investors. They lend/invest and expect a profit/financial reward. The borrowers also known as issuers in the markets borrow and promise to pay the lenders a profit. We therefore encourage savings and investments.

Educating the public about the higher profits in shares and bonds; how to buy and sell; when and why to buy and sell. We also educate the public on how to invest together as a group.

Facilitating good management of companies by asking them to give periodic reports of their performance.

Providing a daily market reports and price list to ensure that investors know the worth of their assets at all times.

Providing financial solutions to common problems. Shares and bonds are accepted guarantees for Co-operative Society’s and bank loans. Shares and bonds can be planned, with the help of a money manager, to pay for school fees, medical, car and other insurance schemes, pension or retirement plans etc.

Through shares and bonds, the government, small and big companies, cooperatives societies and other organizations can raise money to expand their business activities, make a profit, create employment and generally help the economy to grow.

        Q:How is the NSE market organized?
        A:Market days.

The market is open Monday to Friday from 8.00 a.m. to 5.00 p.m. trading activities start at 9.00 a.m. and continues until 3.00 p.m. Members of the public can view the market from the public gallery at any time while the market is open. The market is closed during public holidays.

Display of Shares
Shares are grouped into 4 sectors namely Agriculture, Commercial and Services, Financial and Industrial & Allied sectors. The shares are displayed in alphabetical order in each group for easy location by investors viewing trading from the public gallery.

Display of Bonds
Bonds are in two groups namely: Treasury Bonds – issued by the government; and Corporate Bonds – issued by companies.

They are displayed as and when the government or a company issues one.

        Q:What Amounts Can an Investor Buy?
        A:An investor can buy as little or as much as he or she can afford. It is also possible to invest very little money in groups of small investors pooled together by money managers in the market.

Minimum number of shares.

Share are bundled in minimum lots of 100 shares and above in the main market boards. Fewer shares that 100 are available on the odd lots board.

Minimum number of Bonds.

Bonds are sold in minimum bundles of KShs. 50,000.00. Small investors can pool their money together and buy a bond with the help of a money manager.

        Q:Delivery & Settlement
        A:This is where share accounts and bond accounts are transferred from one investor to another and payments completed. It is also here that shares and bonds of deceased persons are transferred to beneficiaries at a small administrative fee.

        Q:Information Centre
        A:Here, investors and members of the public get answers to their questions about the stock exchange. An individual or group can also register to attend a educational session organized by the Nairobi Securities Exchange every Wednesday. The NSE also gives lectures to groups in their own premises through invitation. There are other operational structures in the market which include Market Research and Development, Compliance, Legal, Accounting and Administration.

        Q:Our History
        A:This market was started in the 1920’s by the British as an informal market for Europeans only. In 1954, the market was formalized through incorporation into a company.

In 1963, Africans were allowed to join and trade in the market. For many years, the market operated through the telephone with a weekly meeting at the Stanley Hotel. In 1991, this market mover to IPS building and was opened to the public. In 1994, the market moved to its current location, on the 1st Floor of the Nation Centre, with the introduction of the Central Depository and Settlement Corporation (CDSC) investors will open share and bond accounts, in electronic accounts similar to their bank accounts. Buying and selling of shares and bonds will be made much easier and quicker. All the benefits of shares and bonds will remain the same. For example, an investor will still be able to use a Share Account or Bond Account as a guarantor for a Co-operative loan or as collateral for a bank loan.

        Q:More About Us.
        A:More information about the NSE can be found at the Nairobi Securities Exchange information centre; website: www.nse.co.ke; Handbook; NSE Annual Report; Brochures; public Education Sessions; Newspapers; Television and Radio. You can also speak to Stockbrokers; Money Managers; Browse Company Annual Reports and Accounts and talk to other investors in the market.

        Q:What Is A Share?
        A:A share is a piece of ownership of a company or enterprise. When you buy a share, you become an investor and thereby an owner of a piece of the company’s profit or losses.

        Q:Why do companies sell shares?
        A:Companies sell shares to raise (borrow) money from members of the public to expand their business activities in order to make more profits. They invite members of the public to buy shares and by so doing have a say in the running of the company as lenders of money and owners.

Shareholders expect a profit as a reward from lending their money to expand the business of the company.

        Q:Who is a shareholder?
        A:A shareholder is an investor who buys shares with an expectation of profit. Profits in shares are through dividends, gains in share prices, bonuses, rights etc.

A shareholder owns a piece of the company and its profits equal to the number of shares he/she owns.

        Q:What are the benefits of owning shares?
        A:A source of profits;
A guarantor for borrowing loans from Cooperative Societies and Banks;
A way of saving your money for the future;
An easy and quick asset to buy and sell;
A new business activity that is beneficial in many ways. An investor can trade in other markets trade in maize, bananas, potatoes, tomatoes, onions, mangos etc.
Buying at low prices and selling at high prices to make a profit;
A solution that increases financial activity and economic growth.

        Q:What are the qualities of a good share?
        A:Frequent and generous dividends
The company is managed productively, transparently and is accountable to shareholders
No wastage in the use of resources
Respect of shareholders and their opinions
Shares that are easy to buy and sell quickly in the market
The company abides by the rules, regulations and laws

        Q:What is a Bond?
        A:A bond is a loan between a borrower and a lender. The borrower promises to pay the lender some interest quarterly or semi-annually at some date in the future. The borrower also promised to repay the initial money invested by the lender. The lender lends and expects to make a profit. The profit from a bond is gained in the form of an interest. At the moment some bonds in the market have an interest rate of 14%, 12%, 10%,8% depending on the type of bond it is, and when it was issued.


        Q:I OWE YOU.
        A:At the Nairobi Securities Exchange, the lender is called an investor and the borrower the issuer.

        Q:Who can Buy Bonds?
        A:Any individual, Co-operative Society, Women Group, Kiama, Youth Club, Church, School, College, University, Investment Group, Insurance Company, Bank, Pension Scheme, and many other can buy bonds.

        Q:Can a Bond be sold before Maturity?
        A:Yes. In times of need or emergencies, an investor can sell his or her bond easily and quickly in the market. The interest on a bond grows on a daily basis and so a bond has new value and price every day. An investor can therefore buy or sell a bond on any day of his or her choice. There are no penalties for selling a bond before the maturity date.

For example, an investor can buy a bond of 5 years and expect an interest of 12%. The interest is paid after every 3 or 6 months. Such an investor can sell the bond at any time of his or her choice at the current market price. The market price of a bond will depend on the number of other willing sellers and buyers in the market on that particular day. When there are many sellers in any market, prices go down and vice versa.

        Q:Who Borrows Money through Bonds?
        A:In Kenya, it is the government and companies. In other Stock Markets, Municipal Councils, Cooperative Societies, Hospitals, Universities, Schools and other organizations can borrow money from the public through bonds. All that is required is that the organization has a good reputation and members of the public have trust in the other lending situations, a lender must trust the borrower before he or she can lend any money. The borrower must therefore be creditworthy in the eyes of the lenders or investors.

Bonds are therefore a very easy, quick and transparent way of raising money. For example, trusted and credit worthy Municipal Councils can borrow money from the public with a promise to pay a reasonable interest rate. The Council can borrow money from public with a promise to pay a reasonable interest rate. The Council can use the money to build roads, improve security, cleanliness, water supply and streetlights. A Co-operative Society can do the same and build milk cooling and processing factory or a food processing factory. These and many more money solutions are available with the help of money managers.

        Q:What are the benefits of buying bonds?
        A:A bond is a very convenient asset to own;
Accepted guarantors for may types of loans by Cooperative Societies and Banks;
A sure source of income;
A good money planner to meet specific needs. For example an investor can buy a bond whose interest matches payment of school fees, car or medical insurance, rent, pension allowance and much more;
Easy and quick to sell in the market in times of need;
A way of saving money for the future;
Convenient and confidential;
Easily transferable.

        Q:What is the difference between a bondholder and a shareholder?
        A:A Bondholder

A bondholder is only a lender to a company
Expects a profit in form of an interest at a specific agreed date in future
Does not vote or participate in the management of the company
Invests to earn a reasonable return at a low risk
A watchdog of the borrowers activities
A Shareholder

A shareholder is a lender and an owner
Expects a profit in form of a dividend, gain in share price, bonuses and cheaper shares (right issues)
Attends Annual General Meetings, gives personal pinions about the company and votes thereby participating in the running of the company
Invests expecting the highest return possible
Accept risk as part of any business
A watchdog of the management and company’s activities
An influencer of the company’s performance
Can one be Bondholder and a Shareholder at the same time?
Yes. This gives an investor the opportunity to diversify and enjoy a balance between reasonable and very high profits.

        Q:What are the eligibility requirements for listing at the Nairobi Securities Exchange
        A:There are three investment market Segments at the Nairobi Securities Exchange namely:

Main Investment Market Segment (MIMS);
Alternative Investment Market Segment (AIMS); and
Fixed Income Securities Market Segment (FISMS).
To list securities on any of these boards, the following eligibility criteria must be satisfied:
Requirements:
1.)Incorporation status
MIMS:The issuer must be a public company limited by shares and registered under the Companies Act (Cap 486)
AIMS:The issuer must be a public company limited by shares and registered under the Companies Act (Cap 486)
FISMS:The issuer must be a public company limited by shares and registered under the Companies Act (Cap 486) or any other corporate body.

2.)Share Capital
MIMS:The minimum authorized, issued and fully paid up capital must be Kshs. 50 Million.
AIMS:The minimum authorized, issued and fully paid capital should be Kshs.20 Million
FISMS:The minimum authorized, issued and fully paid up capital must be Kshs. 50 Million

3.)Net Assets
MIMS:The net assets should not be less than Kshs. 100 Million immediately before the public offer.
AIMS:Net assets immediately before the public offer should not be less than Kshs.20 Million
FISMS:The net assets should not be less than Kshs. 100 Million immediately before the offer.

4.)Financial records
MIMS:The audited financial statements of the issuer for five preceding years be availed.
AIMS:The audited financial statements of the issuer for three preceding years be availed.
FISMS:The audited financial statements of the issuer for three preceding years be availed (except for the government)

5.)Directors and Management
MIMS:The directors of the issuer must be competent persons without any legal encumbrances
AIMS:The directors of the issuer must be competent persons without any legal encumbrances.
FISMS:The directors of the issuer must be competent persons without any legal encumbrances

6.)Track record
MIMS:The issuer must have declared positive profits after tax attributable to shareholders in at least three years within five years prior to application.
AIMS:The issuer must have been operating on the same line of business for at least two years one of which it must have made profit with good growth potential.
FISMS:Not a requirement.

7.)Solvency
MIMS:The issuer should be solvent and have adequate working capital.
AIMS:The issuer should be solvent and have adequate working capital.
FISMS:Not a requirement

8.)Share ownership structure
MIMS:At least 25 percent of the shares must be held by not less than 1000 shareholders excluding employees of the issuer.
AIMS:At least 20 percent of the shares must be held by not less than 100 shareholders excluding employees of the issuer or family members of the controlling shareholders.
FISMS:Not a requirement

9.)Certificate of comfort
MIMS:May be required from the primary regulator of the issuer if there is one.
AIMS:May be required from the primary regulator of the issuer if there is one.
FISMS:May be required from the primary regulator of the issuer if there is one.

10.)Dividend policy
MIMS:The issuer must have a clear future dividend policy
AIMS:The issuer must have a clear future dividend policy
FISMS:Not a requirement

11.)Debt ratios
MIMS:Not a requirement
AIMS:Not a requirement
FISMS:Major ratios:
    Total indebtedness including the new issue not to exceed 400% of the company’s net worth as at the latest balance sheet.
    The funds from operations to total debt for the three trading periods preceding the issue to be kept at a weighted average of at least 40%.
    A range of other ratios to be certified by the issuer’s external auditors.

12.)Issue lots
MIMS:Not a requirement
AIMS:Not a requirement
FISMS:Minimum issue lot size shall be:

Kshs. 100,000 for corporate bonds or preference shares
Kshs. 1,000,000 for commercial paper programme.

13.)Renewal date
MIMS:Not a requirement
AIMS:Not a requirement
FISMS:Every issuer of commercial paper to apply for renewal at least three months before the expiry of the approved period of twelve months from the date of approval.

14.)Transferability of shares
MIMS:The shares to be listed shall be freely transferable.
AIMS:The shares to be listed shall be freely transferable.
FISMS:May or may not be transferable.






        Q:Can one be Bondholder and a Shareholder at the same time?
        A:Yes. This gives an investor the opportunity to diversify and enjoy a balance between reasonable and very high profits.

        Q:What is a Real Estate Investment Trust (REIT)
        A:A REIT is a regulated investment vehicle that enables the issuer to pool investors’ funds for the purpose of investing in real estate. In exchange, the investors receive units in the trust, and as beneficiaries of the trust,share in the profits or income from the real estate assets owned by the
trust.

        Q:What types of REIT's are provided for in Kenya?
        A:There are 2 types of REITS; D-REITS and I-REITS.

Income Real estate investment Trusts (I-REITs)
An Income-REIT (I-REIT) is a real estate investment scheme which owns and manages income generating real estate for the benefit of its investors therefore providing both liquidity and a stable income stream. Distributions to investors are underpinned by commercial leases. This
means that income returns are generally predictable.

Development Real Estate Investment Trusts (D-REITs)
A Development Real Estate Investment Trust (D-REIT) is a development and construction real estate trust involved in the development or construction projects for housing, commercial and other real estate assets.

        Q:What is an Exchange Traded Fund(ETF) ?
        A:An Exchange Traded Fund (ETF) is a listed investment product that track the performance of a basket of Shares, Bonds or Commodities. An ETF can also track a single commodity such as oil or a precious metal like gold.

        Q:Why invest in ETFs ?
        A:1. Diversification
One fund can hold potentially hundreds—sometimes thousands—of individual stocks and bonds, which helps spread out risk. ETFs allow access to asssets that were previously not available to all investors such as gold.
2. Professional management
You do not have to keep track of every single investment your ETF owns. The fund is managed by fund experts who take care of that for you. Reporting is done daily to the investor through the NSE.
3.Liquidity
ETFs offer you the same liquidity you get when trading stocks and bonds listed on the NSE.
4.Lower cost
Funds that track an index, like ETFs and index mutual funds, generally offer lower expense ratios than conventional mutual funds.
5.Transparency
The ETF is backed by its constituent underlying assets or assets of an equivalent value.
Issuers produce a factsheet for their ETFs which states what investors are being exposed to and how the Net Asset Value of the ETF is calculated. By contrast, unit trusts, typically only provide historical information on portfolio holdings and not current holdings, for competitive reasons. Tracking performance is also published.
6.Investor owned assets
The constituent assets or securities shall be housed in a trust arrangement with a CMA approved trustee being appointed. Even in the case of insolvency by the ETF manager, administrator or issuer, through the trust arrangement, these assets are ring fenced, protected by law, and are the exclusive property of the ETF. Therefore owning an ETF does not give the investor the right to vote at Annual General Meetings (AGMs) of the underlying securities, as you own a portion (unit) in the fund and not the underlying securities themselves.
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
        # Extract the header to keep context in every chunk
        header_match = text.split('\n\n')[0] if "DOCUMENT:" in text[:100] else ""
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + (chunk_size // 2):
                    end = last_period + 1
            
            chunk_content = text[start:end]
            
            # Inject header into EVERY chunk if it's missing
            if header_match and header_match not in chunk_content:
                final_chunk = f"{header_match}\n[FRAGMENT]\n{chunk_content}"
            else:
                final_chunk = chunk_content
                
            chunks.append(final_chunk)
            start += chunk_size - overlap
            if start >= end: start = end
        return chunks

    def _extract_text_from_pdf(self, pdf_content):
        """
        Preserves Table structures as Markdown using pdfplumber if available.
        """
        try:
            import pdfplumber 
            
            text_content = []
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    page_text = page.extract_text()
                    
                    if tables:
                        for table in tables:
                            clean_table = [[str(cell).replace('\n', ' ') if cell else "" for cell in row] for row in table]
                            if len(clean_table) > 0:
                                headers = clean_table[0]
                                header_row = "| " + " | ".join(headers) + " |"
                                separator = "| " + " | ".join(["---"] * len(headers)) + " |"
                                body = "\n".join(["| " + " | ".join(row) + " |" for row in clean_table[1:]])
                                table_md = f"\n\n[TABLE_DATA_PAGE_{i+1}]\n{header_row}\n{separator}\n{body}\n\n"
                                text_content.append(table_md)

                    if page_text:
                        text_content.append(f"\n[PAGE {i+1}] {page_text}")
                        
            return "\n".join(text_content)
            
        except ImportError:
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
        except Exception as e:
            print(f"PDF Error: {e}")
            return ""

    def _get_random_header(self):
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        ]
        return {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_url(self, url):
        time.sleep(random.uniform(0.5, 1.5)) 
        return self.session.get(url, headers=self._get_random_header(), timeout=25, verify=False)

    def crawl_site(self, seed_urls):
        visited = set()
        to_visit = set(seed_urls)
        found_content_urls = set() 
        found_pdf_urls = set()     
        
        # You can paste your specific hardcoded links here
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
                            # Limit queue size to prevent memory issues
                            if len(to_visit) < 1000: 
                                to_visit.add(full_url)
                                
            except Exception as e:
                print(f"Crawl Error {url}: {e}")
        
        all_pdfs = list(found_pdf_urls.union(set(hardcoded_pdfs)))
        return list(found_content_urls), all_pdfs

    def _process_content(self, url, content_type, content_bytes):
        text = ""
        tag = "[GENERAL]"
        
        # --- TAGGING SYSTEM ---
        if "faq" in url.lower(): tag = "[OFFICIAL_FAQ]"
        elif "statistics" in url: tag = "[MARKET_DATA]"
        elif "leadership" in url or "board" in url: tag = "[LEADERSHIP]"
        elif "rules" in url: tag = "[REGULATION]"
        elif "financial" in url: tag = "[FINANCIALS]"
        elif "etf" in url or "bond" in url: tag = "[PRODUCT]"

        filename = url.split('/')[-1].replace('.pdf', '').replace('-', ' ').title()

        if content_type == "pdf":
            raw_text = self._extract_text_from_pdf(content_bytes)
            if len(raw_text) > 100:
                header = f"DOCUMENT: {filename}\nSOURCE: {url}\nTAGS: {tag}\nTYPE: OFFICIAL PDF\n\n"
                text = header + self.clean_text_chunk(raw_text)
        else:
            soup = BeautifulSoup(content_bytes, 'html.parser')
            for item in soup(["script", "style", "nav", "footer", "header", "aside"]):
                item.decompose()
            raw_text = soup.get_text(separator="\n")
            clean = self.clean_text_chunk(raw_text)
            if len(clean) > 100:
                header = f"DOCUMENT: {filename}\nSOURCE: {url}\nTAGS: {tag}\nTYPE: Webpage\n\n"
                text = header + clean
        
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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

    def enrich_query_with_tickers(self, query):
        ticker_map = {
            "safaricom": "SCOM",
            "equity": "EQTY", "equity group": "EQTY", "equity bank": "EQTY",
            "kcb": "KCB", "kcb group": "KCB",
            "co-op": "COOP", "cooperative bank": "COOP", "coop": "COOP",
            "absa": "ABSA", "barclays": "ABSA",
            "eabl": "EABL", "breweries": "EABL",
            "kenya power": "KPLC", "kplc": "KPLC",
            "kenya airways": "KQ", "kq": "KQ",
            "standard chartered": "SCBK", "stanchart": "SCBK",
            "ncba": "NCBA",
            "britam": "BRIT",
            "cic": "CIC",
            "centum": "CTUM",
            "total": "TOTL", "totalenergies": "TOTL"
        }
        
        lower_query = query.lower()
        found_tickers = []
        for name, ticker in ticker_map.items():
            if name in lower_query and ticker not in lower_query.upper():
                found_tickers.append(ticker)
        
        if found_tickers:
            return f"{query} ({' '.join(found_tickers)})"
        return query

    def _extract_keywords(self, query):
        """
        NEW: Extracts core keywords (removing stop words) to check against URLs.
        """
        stops = set(["the", "is", "at", "which", "on", "and", "a", "an", "of", "for", "in", "to", "how", "what", "where", "does", "about"])
        # Simple regex to get words > 2 chars
        words = re.findall(r'\w+', query.lower())
        return [w for w in words if w not in stops and len(w) > 2]

    def generate_context_queries(self, original_query):
        today = datetime.date.today().strftime("%Y-%m-%d")
        enriched_query = self.enrich_query_with_tickers(original_query)
        
        prompt = f"""Generate 3 search queries for the NSE database based on: "{enriched_query}"
        RULES:
        1. Append "NSE" or "Nairobi Securities Exchange".
        2. Output ONLY the 3 queries, one per line.
        """
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3
            )
            return [q.strip() for q in response.choices[0].message.content.split('\n') if q.strip()][:3]
        except:
            return [enriched_query + " NSE", "NSE Kenya " + enriched_query]

    def llm_rerank(self, query, documents, sources):
        if not documents: return []
        
        candidates = ""
        for i, doc in enumerate(documents):
            candidates += f"\n--- DOC {i} ---\nSource: {sources[i]}\nContent: {doc[:300]}...\n"
            
        prompt = f"""Rank these documents by relevance to: "{query}" (Context: NSE Kenya).
        Return IDs of top 5 (e.g., "0, 3, 1").
        
        CRITICAL RANKING RULES:
        1. If a document has the tag [OFFICIAL_FAQ], it MUST be ranked #1.
        2. If the Source URL contains exact keywords from the query, Rank it HIGH.
        3. Prioritize [LEADERSHIP] or [OFFICIAL PDF] tags.
        
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

            search_queries = self.generate_context_queries(query)
            
            # Retrieve (Broaden search to find URL matches that might have low vector scores)
            query_embeddings = self.get_embeddings_batch(search_queries)
            results = self.collection.query(query_embeddings=query_embeddings, n_results=25)
            
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
            
            # --- NEW: KEYWORD URL BOOSTING ---
            # This sorts the raw results *before* the LLM sees them.
            # We verify if the URL contains key terms from the user's query.
            keywords = self._extract_keywords(query)
            scored_results = []
            
            for doc, source in zip(raw_docs, raw_sources):
                score = 0
                source_lower = source.lower()
                
                # Boost 1: URL Match (The feature you requested)
                for kw in keywords:
                    if kw in source_lower:
                        score += 10 # Significant boost
                
                # Boost 2: FAQ Priority
                if "faq" in source_lower:
                    score += 20
                
                # Boost 3: PDFs are usually better than random HTML pages
                if ".pdf" in source_lower:
                    score += 2
                    
                scored_results.append((score, doc, source))
            
            # Sort by calculated score (descending), then pass to LLM
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Extract back to lists, taking top 15 best candidates
            sorted_docs = [x[1] for x in scored_results[:15]]
            sorted_sources = [x[2] for x in scored_results[:15]]

            # Re-rank with LLM (Now the LLM sees the URL-matched docs first)
            top_results = self.llm_rerank(query, sorted_docs, sorted_sources)
            
            # Final Context Construction
            context_text = self.get_static_facts() + "\n\n--- RETRIEVED DATA ---\n"
            visible_sources = []
            for doc, source in top_results[:5]: 
                context_text += f"\n[Source: {source}]\n{doc}\n---"
                if source not in visible_sources: visible_sources.append(source)

            today = datetime.date.today().strftime("%Y-%m-%d")
            
            system_prompt = f"""You are the NSE Digital Assistant.

            MANDATORY CONTEXT INTERPRETATION:
            You must treat the user's query as if it ends with "...in the context of the Nairobi Securities Exchange (NSE)".
            
            POWER RULES:
            1. **FAQ Authority:** If the context contains text tagged [OFFICIAL_FAQ], that is the GROUND TRUTH. Answer using that information exactly as stated.
            2. **URL Relevance:** Documents where the Source URL matches the user's keywords are likely the most correct.
            3. **Financial Responsibility:** NEVER provide investment advice (Buy/Sell/Hold).
            4. **Tabular Data:** If the query asks for financial performance, format as a Markdown Table.
            5. **Fact Sheet Priority:** Use the [OFFICIAL_FACT_SHEET] for Leadership/Hours/Market Structure.
            
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