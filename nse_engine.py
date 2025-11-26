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
MAX_PAGES_TO_CRAWL = 1000
PINECONE_INDEX_NAME = "nse-data"
PINECONE_DIMENSION = 1536 

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
            try:
                self.pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(10) # Wait for init
            except Exception as e:
                print(f"Index creation warning: {e}")
            
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.session = requests.Session()

    # ... (Rest of the engine logic: get_static_facts, get_embeddings_batch, get_embedding, build_knowledge_base, scrape_and_upload, generate_context_queries, answer_question, helper methods) ...
    # (This part is identical to the previously provided complete nse_engine.py code. Ensure you copy the full content from previous responses if needed)

    # --- STATIC KNOWLEDGE ---
    def get_static_facts(self):
        return """
        [OFFICIAL_FACT_SHEET]
        TOPIC: NSE Leadership, Structure & Market Rules
        SOURCE: NSE Official Website / Annual Report 2025
        LAST_VERIFIED: November 2025

        CEO: Mr. Frank Mwiti
        Chairman: Mr. Kiprono Kittony
        Non-Executive Director representing listed companies: Ms. Risper Alaro-Mukoto
        Independent Non-Executive Director: Ms. Isis Nyong'o Madison
        Non-Executive Director representing Trading Participants: Mr. Donald Wangunyu
        Non-Executive Director representing Listed Companies: Mr. Stephen Chege
        Independent Non-Executive Director :Mr. John Niepold
        Independent Non-Executive Director: Ms. Carole Kariuki
        Independent Non-Executive Director: Mr. Thomas Mulwa
        
        Trading Hours:(Monday - Friday, Excluding Public Holidays)
    - Pre-Open Session: 09:00 AM - 09:30 AM
    - Continuous Trading: 09:30 AM - 03:00 PM
    - Closing Session: 03:00 PM onwards
    Pre-Open Session: 09:00 am - 09:30 am

        Currency: Kenyan Shilling (KES)
        Regulator: Capital Markets Authority (CMA)
        Depository: Central Depository & Settlement Corporation (CDSC)
        Key Indices: NSE All Share Index (NASI), NSE 20 Share Index, NSE 25 Share Index
        Settlement Cycle: T+3 (Equities), T+3 (Corporate Bonds), T+1 (Gov Bonds)
        Location: The Exchange, 55 Westlands Road, Nairobi, Kenya
        
        
        2. OPERATIONAL DETAILS
        - Website: www.nse.co.ke

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
        [OFFICIAL_NSE_20_SHARE_INDEX_RULES_V1.6_MAY_2025]
SOURCE: https://www.nse.co.ke/rules/ – Ground Rules for the Management of the NSE 20 Share Index v1.6 (Approved 21/05/2025)

The NSE 20 Share Index is a market capitalization-weighted index consisting of 20 of the most liquid and largest companies listed on the Nairobi Securities Exchange.

Objective: The objective of the NSE 20 Share Index is to reflect the performance of the top 20 companies by market capitalization and liquidity.

Selection Criteria:
1. The company must be listed on the Main Investment Market Segment (MIMS) or the Growth Enterprise Market Segment (GEMS) of the Nairobi Securities Exchange.
2. The company must have a minimum free-float market capitalization of KES 1 billion.
3. The company must have a minimum average daily trading value of KES 5 million over the previous six months.
4. The company must have traded on at least 70% of the trading days over the previous six months.

Index Calculation: The NSE 20 Share Index is calculated using the Laspeyres formula with market capitalization weighting. The index is price-return only (dividends are not reinvested).

Review Frequency: The index constituents are reviewed semi-annually in May and November each year.

Capping: No single constituent shall exceed 15% of the total index weight. If a constituent exceeds 15% at review, its weight will be capped at 15% and the excess weight redistributed proportionally among the remaining constituents.

Buffering: To minimize turnover, a buffer rule is applied. Existing constituents ranked 25 or better remain in the index, while non-constituents must rank 15 or better to enter.

Corporate Actions Treatment:
- Bonus issues, stock splits, and rights issues are adjusted by changing the number of shares without affecting the price.
- Cash dividends are not adjusted (price-return index).
- In case of delisting, suspension, or merger, the company is removed from the index at the earliest opportunity.

Current Constituents (as at 21 May 2025):
1. Safaricom Plc
2. Equity Group Holdings Plc
3. KCB Group Plc
4. East African Breweries Plc
5. ABSA Bank Kenya Plc
6. Co-operative Bank of Kenya Plc
7. NCBA Group Plc
8. Standard Chartered Bank Kenya Ltd
9. BK Group Plc (Bank of Kigali)
10. I&M Group Plc
11. CIC Insurance Group Plc
12. Britam Holdings Plc
13. Kenya Reinsurance Corporation Ltd
14. Jubilee Holdings Ltd
15. Centum Investment Company Plc
16. B.O.C Kenya Plc
17. Kakuzi Plc
18. HF Group Plc
19. Diamond Trust Bank Kenya Ltd
20. Williamson Tea Kenya Plc

Latest Changes (21 May 2025):
- Bamburi Cement Plc removed
- Nation Media Group Plc removed
- HF Group Plc added
- Diamond Trust Bank Kenya Ltd added

The index is calculated and disseminated in real-time during trading hours. The base date is 1st February 2008 with a base value of 5,000 points.

This document is reviewed and approved by the NSE Trading Committee. The latest version is always available at https://www.nse.co.ke/rules/



[OFFICIAL_NSE_BANKING_SECTOR_SHARE_INDEX_RULES_V1.0_SEPTEMBER_2025]
SOURCE: NSE Official Document – Ground Rules for the Management of the NSE Banking Sector Share Index v1.0 (Approved 01/09/2025)

The NSE Banking Sector Share Index is a market capitalization-weighted index that tracks the performance of all licensed commercial banks and banking groups listed on the Nairobi Securities Exchange.

Objective: To provide a benchmark for the performance of the banking sector in Kenya and serve as an investable index for financial products.

Index Constituents (as at 1st September 2025):
1. ABSA Bank Kenya Plc
2. BK Group Plc (Bank of Kigali)
3. Co-operative Bank of Kenya Plc
4. Diamond Trust Bank Kenya Ltd
5. Equity Group Holdings Plc
6. HF Group Plc
7. I&M Group Plc
8. KCB Group Plc
9. NCBA Group Plc
10. Standard Chartered Bank Kenya Ltd

Index Calculation Methodology:
- The index is calculated using the Laspeyres price index formula with free-float market capitalization weighting.
- It is a price-return index (dividends are not included).
- Base Date: 3rd January 2022
- Base Value: 1,000 points

Eligibility Criteria:
All companies classified under the "Banking" sub-sector by the Nairobi Securities Exchange and licensed by the Central Bank of Kenya are automatically included in the index.

Weighting and Capping:
- Weights are based on free-float market capitalization.
- No single constituent shall exceed 25% of the total index weight.
- If a constituent exceeds 25% during quarterly reviews, its weight is capped at 25% and the excess redistributed proportionally among remaining constituents.

Review Frequency:
The index is reviewed quarterly (March, June, September, December) with changes implemented on the first trading day following the third Friday of the review month.

Corporate Actions:
- Bonus issues, stock splits, and rights issues are adjusted by modifying the number of shares.
- Cash dividends are not adjusted (price-return index).
- In case of delisting, merger, or acquisition, the affected company is removed and weights redistributed.

Publication:
The index is calculated and disseminated in real-time during trading hours. End-of-day values are published on the NSE website and data services.

This document was drafted by the Operations Department, reviewed by Trading Data & Analytics, and approved by the Trading Committee on 1st September 2025.

[OFFICIAL_EAE_20_SHARE_INDEX_RULES_2025]

SOURCE: East African Securities Exchanges Association (EASEA) – Ground Rules for the Management of the EAE 20 Share Index (2025)

The EAE 20 Share Index is a market capitalization-weighted index consisting of 20 of the largest and most liquid blue-chip companies listed on the participating East African securities exchanges (Nairobi Securities Exchange, Dar es Salaam Stock Exchange, Rwanda Stock Exchange, Uganda Securities Exchange).

Objective: The index reflects the total market value of component stocks relative to the base period and serves as a barometer of market performance across the East African region.

Base Date: 7th March 2025
Base Value: 100

Eligibility Criteria:
1. The company’s shares must be primarily listed on one of the participating East African securities exchanges.
2. The company must have maintained a continuous listing for a minimum of one year on the respective exchange.
3. The company should be a recognized blue-chip firm, demonstrating strong profitability and a consistent dividend payment history.

Current Constituents (as at 2025):
1. CRDB Bank
2. NMB Bank
3. Tanzania Breweries Limited
4. Tanga Cement Company Limited
5. Tanzania Cigarette Corporation
6. Bralirwa Limited
7. Bank of Kigali
8. I&M Rwanda
9. Cimerwa Plc
10. MTN Rwandacell
11. Safaricom Plc
12. KCB Group Plc
13. Equity Group Holdings Plc
14. The Co-operative Bank of Kenya Ltd
15. ABSA Bank Kenya Plc
16. MTN Uganda
17. Stanbic Uganda Holdings
18. Bank of Baroda Uganda
19. Airtel Uganda
20. Quality Chemicals Industry Limited

Review: Semi-annual by the EASEA Secretariat. Reserve list of the five highest-ranking non-constituents is published after each review.

Corporate Actions: Adjustments are made for delistings, suspensions, mergers, new issues, etc. The index uses the Laspeyres formula with market capitalization weighting.

The index is calculated and disseminated in real-time during trading hours by the member exchanges.


[OFFICIAL_NSE_25_SHARE_INDEX_RULES_V1.4_MAY_2025]
SOURCE: Nairobi Securities Exchange – Ground Rules for the Management of the NSE 25 Share Index v1.4 (Approved 21/05/2025)

The NSE 25 Share Index represents the performance of the top 25 Kenyan companies listed on the Nairobi Securities Exchange.

Eligibility Criteria:
i. Shares must have their primary listing on the Nairobi Securities Exchange.
ii. Must have at least 20% of its shares quoted on the NSE.
iii. Must have been continuously quoted for at least 1 year.
iv. Must have a minimum market capitalization of Kes.1 billion.
v. Should ideally be a “blue chip” with superior profitability and dividend record.

Current Constituents (as at 21 May 2025):
1. ABSA Bank Kenya Plc
2. Stanbic Holdings Plc
3. Diamond Trust Bank Kenya Ltd
4. Equity Group Holdings Plc
5. I&M Holdings Plc
6. KCB Group Plc
7. NCBA Group Plc
8. Standard Chartered Bank Kenya Ltd
9. The Co-operative Bank of Kenya Ltd
10. BK Group Plc
11. Trans-Century Plc
12. Jubilee Holdings Ltd
13. HF Group Plc
14. KenGen Co. Plc
15. Carbacid Investments Plc
16. Kenya Power & Lighting Co Plc
17. Britam Holdings Plc
18. CIC Insurance Group Ltd
19. Kenya Re Insurance Corporation Ltd
20. Liberty Kenya Holdings Ltd
21. Centum Investment Co Plc
22. Nairobi Securities Exchange Plc
23. British American Tobacco Kenya Plc
24. East African Breweries Ltd
25. Safaricom Plc

Review: Semi-annual (last week of May and November). Ranking by market capitalization with buffering rules.

Latest Changes (21 May 2025): Replaced Bamburi Cement Plc, Nation Media Group Plc, WPP Scangroup Plc and Total Kenya Ltd with HF Group Plc, BK Group Plc, Trans-Century Plc and Carbacid Investments Plc respectively.

The index is calculated using market capitalization weighting and disseminated in real-time.

[OFFICIAL_NSE_10_SHARE_INDEX_RULES_V2.0_MAY_2025]
SOURCE: Nairobi Securities Exchange – Ground Rules for the Management of the NSE 10 Share Index v2.0 (Approved 21/05/2025)

The NSE 10 Share Index (N10) represents the performance of the 10 most liquid stocks listed on the Nairobi Securities Exchange.

Eligibility & Selection: Top 10 companies screened by liquidity measures – Market capitalization (float adjusted) 40%, Turnover 30%, Volume 20%, Deals 10%. Reviewed semi-annually.

Base Date: 30th August 2023
Base Value: 1000

Current Constituents (as at 21 May 2025):
1. ABSA Bank Kenya Plc
2. Equity Group Holdings Plc
3. KCB Group Plc
4. The Co-operative Bank of Kenya Ltd
5. Kenya Re Insurance Corporation Ltd
6. HF Group Plc
7. KenGen Co. Plc
8. East African Breweries Ltd
9. I&M Group Plc
10. Safaricom Plc

Calculation: Base-weighted aggregate methodology (market capitalization/value weighted, float adjusted). Formula: NSE10 = (Current Market Value / Base Market Value) × 1000

The index is reviewed during the last week of every calendar year with divisor adjustments for corporate actions.


[OFFICIAL_NSE_FIXED_INCOME_TRADING_RULES_JANUARY_2024]
SOURCE: Nairobi Securities Exchange – Trading Rules for Fixed Income Securities (January 2024)

The Fixed Income Securities Market operates as a Hybrid Bond Market Structure combining Onscreen Trading and OTC Trading.

Trading Hours: Daily sessions from 9:00 a.m. to 3:00 p.m.

Boards:
- Unrestricted Board: Publicly offered listed fixed income securities.
- Restricted Board: Securities from restricted public offers (sophisticated investors only).

Settlement: T+3 for all fixed income securities transactions.

Quotations Board: Authorized Participants must provide two-way indicative quotes for Benchmark Bonds before 9:00 a.m. Spread not exceeding 100 basis points.

Order Input & Execution:
- Onscreen: Price-time priority matching in ATRS.
- OTC: Bilateral trades reported within 30 minutes to the Central Trade Repository.

Board Lot: Minimum as prescribed by issuer, not less than one unit.

Sell Buy Backs: Must be pre-agreed in writing and reported.

Trading Halts: Market or security halts may be imposed by the Chief Executive in consultation with the Authority.

Reference Price: Volume-weighted yield of trades > KES 50 million; otherwise based on indicative yields.

All listed fixed income securities are traded through the ATRS unless exempted. English is the official language. Disaster recovery provisions apply.


[OFFICIAL_NSE_NOMINATED_ADVISORS_RULES_2012]
SOURCE: Nairobi Securities Exchange – Nominated Advisors (NomAds) Rules, 2012 (as amended)

A Nominated Advisor (NomAd) is an advisory firm approved by the Nairobi Securities Exchange to guide and advise companies listed or seeking listing on the Growth Enterprise Market Segment (GEMS).

Eligibility Criteria for NomAds:
- Must be a corporation licensed by the Capital Markets Authority as an investment bank, stockbroker or fund manager.
- Must have at least two key professionals with a minimum of five years relevant experience in corporate finance.
- Must demonstrate capacity to advise on CMA and NSE regulations.
- Must not have been censured by CMA or NSE in the last three years.

Key Responsibilities of NomAds:
- Advise the issuer on compliance with GEMS continuing listing obligations.
- Ensure all announcements and disclosures are accurate and timely.
- Guide the issuer on corporate governance best practices.
- Confirm to the NSE that the issuer is suitable for GEMS listing.
- Act as the primary point of contact between the issuer and the NSE.
- Monitor the issuer’s compliance and report any breaches immediately.

NomAd Appointment:
- Every GEMS issuer must appoint and retain a NomAd at all times.
- The appointment must be formalised through a written agreement.
- Any change of NomAd must be notified to the NSE within 14 days.

Disciplinary Actions:
The NSE may suspend or cancel NomAd approval for misconduct, incompetence or failure to supervise issuers properly.

NomAds are jointly liable with the issuer for any false or misleading statements in listing documents.


[OFFICIAL_NSE_MARKET_PARTICIPANTS_RULES_2014]
SOURCE: Nairobi Securities Exchange – Market Participants Rules, 2014 (as amended)

These rules govern the conduct of all Trading Participants (stockbrokers), Authorized Securities Dealers, and other market participants.

Categories of Trading Participants:
1. Full Trading Participant – may trade equities, bonds and derivatives.
2. Fixed Income Trading Participant – restricted to bonds only.
3. Derivatives Trading Participant – restricted to derivatives only.

Eligibility for Trading Participant License:
- Must be licensed by CMA as a stockbroker or investment bank.
- Minimum paid-up share capital of KES 50 million (full participant).
- At least two qualified traders with NSE/CIS certification.
- Adequate systems, risk management and business continuity plans.

Key Obligations:
- Act honestly, fairly and in the best interests of clients.
- Maintain minimum liquid capital requirements at all times.
- Segregate client funds and securities.
- Execute client orders on best execution terms.
- Report all trades accurately and within required timelines.
- Cooperate fully with NSE surveillance and investigations.

Disciplinary Actions:
The NSE Disciplinary Committee may impose fines, suspension or termination for breaches including insider trading, market manipulation, failure to settle, or inadequate capital.

All Trading Participants must contribute to the NSE Fidelity Fund for investor protection.

[OFFICIAL_NSE_LISTING_RULES_2025]
SOURCE: Nairobi Securities Exchange Limited – Listing Rules (as amended to November 2025)

Market Segments:
1. Main Investment Market Segment (MIMS) – large established companies.
2. Alternative Investment Market Segment (AIMS) – medium-sized companies.
3. Growth Enterprise Market Segment (GEMS) – SMEs and high-growth companies.
4. Fixed Income Securities Market Segment (FIMS) – corporate and government bonds.

MIMS Equity Listing Requirements:
- Minimum paid-up capital KES 500 million.
- Minimum 25% public float (at least 1 million shares in public hands).
- Profitable for at least 3 of the last 5 years.
- Minimum 1,000 shareholders.

AIMS Equity Listing Requirements:
- Minimum paid-up capital KES 100 million.
- Minimum 20% public float.
- No profitability track record required but must demonstrate growth potential.

GEMS Listing Requirements:
- Minimum paid-up capital KES 50 million.
- Minimum 10% public float.
- Must appoint a Nominated Advisor (NomAd) at all times.
- Fast-track listing within 3 months.

Continuing Obligations (All Segments):
- Timely disclosure of price-sensitive information.
- Annual and half-year financial reporting.
- Minimum 10% public float maintained.
- Corporate governance code compliance.
- Immediate notification of board changes, substantial transactions, or related party transactions.

Corporate Bonds Listing Requirements:
- Issuer credit rating or guarantee required.
- Minimum issue size KES 300 million.
- Trustee appointment mandatory.

The NSE reserves the right to suspend or delist securities for non-compliance.


[OFFICIAL_NSE_DERIVATIVES_RULES_JULY_2017_AS_AMENDED_2025]
SOURCE: Nairobi Securities Exchange – NSE Derivatives (NEXT) Rules, July 2017 (as amended to November 2025)

The NEXT Derivatives Market offers single stock futures, index futures and options.

Eligible Participants:
- Derivatives Trading Participants (licensed by CMA and NSE).
- Clients must sign risk disclosure and suitability acknowledgements.

Contracts Currently Available:
- Single Stock Futures on Safaricom, Equity Group, KCB Group, EABL, ABSA Kenya, Co-op Bank, StanChart Kenya.
- NSE 25 Share Index Futures.

Contract Specifications:
- Contract size: 100 shares per contract.
- Quotation: In Kenyan Shillings per share/index point.
- Minimum price movement: KES 0.05 (futures), KES 0.01 (options).
- Settlement: Cash settlement on expiry.
- Expiry: Third Thursday of March, June, September, December.

Trading Hours: 10:00 a.m. to 3:00 p.m.

Margins:
- Initial margin set by NSE Clear based on volatility (SPAN methodology).
- Variation margin paid daily.
- Intra-day margin calls if required.

Position Limits:
- Maximum 20% of open interest per client per contract.
- Higher limits possible with NSE approval.

Market Makers:
Appointed market makers must provide continuous two-way quotes.

Default Handling:
NSE Clear acts as central counterparty. Defaults handled via close-out, use of margins, fidelity fund and penalties.

All trades are cleared and guaranteed by NSE Clear Limited.

[NSE Clear Status on IOSCO PFMI Principles]
Source File: nse-clear-status-on-pfmi-principles-april-2021.pdf
Legal Basis (Principle 1):

NSE Clear is licensed under the Capital Markets Act and the Capital Markets (Derivatives Markets) Regulations 2015.

NSE Derivatives Rules (approved by CMA in 2016) govern all aspects of mandated responsibilities.

Governance (Principle 2):

NSE Clear was established in 2014 as the Central Counterparty (CCP).

It has a separate Board of Directors.

Key committees include the Oversight Committee, Advisory Committee, and Risk Management Committee.

Margin (Principle 6):

NSE Clear calculates margin through the Historical Value-at-Risk (VaR) methodology.

Settlement Finality (Principle 8):

The Financial Market Infrastructure (FMI) provides clear and certain final settlement.

Money Settlements (Principle 9):

NSE Clear uses prefunded money in the commercial bank system for settlement purposes.

Segregation and Portability (Principle 14):

The CCP has rules and procedures that enable the segregation and portability of positions of a participant’s customers and the collateral provided to the CCP with respect to those positions.

Custody and Investment Risks (Principle 16):

The FMI safeguards its own and its participants’ assets and minimizes the risk of loss on and delay in access to these assets.


Mark-to-Market MethodologySource File: mark-to-market-methodology-april-2021.pdfOverview:All futures positions are marked-to-market based on the daily settlement price at the end of each trading day7.Profits/losses are computed as the difference between the trade price (or previous day's settlement price) and the current day's settlement price8.Settlement Price Methodologies:Volume Weighted Average Price (VWAP): Preferred for liquid contracts. Calculated as Total Value Traded divided by Total Volume Traded9.Theoretical Price (Spot Price + Cost of Carry): Used if there are no trades or the contract is illiquid.Formula: $F = S \times e^{(r-y)t}$Where $F$ = Futures price, $S$ = Spot price, $r$ = Risk-free rate, $y$ = Dividend yield, $t$ = Time to maturity10.Spot Price Definition: The closing price of the underlying security as published by the Exchange11.Settlement of Profits/Losses:Pay-in and pay-out of mark-to-market settlement amounts is done on T+112.Final Settlement Price:On expiry, positions are marked to the final settlement price (closing price of the underlying asset on expiry day) and settled in cash


 [Corporate Action Handling Guide]
 Source File: corporate-action-management-guide.pdf
 Definition: A Corporate Action is an action that brings material change to a company and impacts shareholders14.Types of Corporate Actions Considered:Special Dividends: One-time distribution of earnings from exceptional profits15.Bonus/Script Issue: Awarding additional securities free of payment16.Rights Issue: Issue of new ordinary shares to existing shareholders at a discounted price17.Mergers & Acquisitions: Combination of companies where shares are exchanged18.Stock Splits/Reverse Splits: Splitting or consolidating shares according to a set ratio19.Adjustment Formulas:Special Dividends: Adjustment Factor (AF) = $(P_{cum} - D) / P_{cum}$20.Bonus Issue: $AF = O / (O + N)$ where O is old shares and N is new shares21.Rights Issue: $AF = P_{ex} / P_{cum}$ (Theoretical ex-right price / Cum-right price)22.Stock Split: $AF = O / N$23.Mergers: If liquid, old instruments are replaced with new ones using ratio $O/N$24.

 
 [Margin Calculation Methodology]
 Source File: initial-margin-calculation-methodology.pdf
 Methodology: Initial Margins (IMs) are calculated based on the Historical Value at Risk (VaR) methodology25.Parameters:Historical Data: 750 trading days (3 years) of spot market data is used26.Daily Returns: Calculated as the natural log of daily returns ($ln(P_t / P_{t-1})$)27.Confidence Interval: A conservative confidence interval of 99.95% is used28.VaR Calculation: Rank returns from largest to smallest and pick the return falling in the 99.95 percentile29.Initial Margin Formula: $IM = Contract Size \times VaR \times Current Market Price$30.Liquidation Period Scale Up:The Exchange uses a two-day liquidation period.IM is scaled by $\sqrt{2}$ ($IM \times \sqrt{2}$)31.Further-Dated Contracts: Calculated using a 50% confidence interval to determine the increment between contract expiries

[NSE Derivatives Market Membership Criteria & Fees]
Source File: nse-derivatives-membership-criteria-and-fees-february-2021.pdf

Membership Categories:

Clearing Member (CM): Performs clearing and settlement for the market. Requires KES 1 Billion Net Worth.

Trading Member (TM): Trades on behalf of clients or own proprietary account. Requires 13 weeks operating costs capital adequacy.

Trading Member (Proprietary): Trades only for own proprietary account. Requires 10 weeks operating costs capital adequacy.


Non-Executing Member (Custodians): Accepts trades executed on behalf of custodial clients to facilitate settlement.

Fees:

Clearing Member: Joining KES 500,000; Annual KES 100,000.

Trading Member: Joining KES 100,000; Annual KES 100,000.

Proprietary TM: Joining KES 50,000; Annual KES 50,000.

Non-Executing Member: Joining KES 100,000; Annual KES 100,000.


Application Documents: Cover letter, Certificate of Incorporation, Company PIN, CR12, etc..

[Default Handling Procedure]
Source File: default-handling-procedure.pdf
Scope: Covers defaults by Clients, Trading Members (TM), and Clearing Members (CM).
Client Default: Typically handled by the TM who performed KYC and collects margin. The TM is required to stand good for their direct clients.
TM Default: Handled by the CM. The CM performs KYC on the TM and collects margin.


CM Default: Handled by the Exchange/Clearing House (NSE Clear).

Settlement Guarantee Fund (SGF) Usage Order:
Defaulter's Collateral.
Defaulter's SGF Contribution.
NSE Clear's Skin-in-the-Game (25% of SGF contribution).
Non-defaulting Members' SGF Contributions.
NSE Clear Equity.
Liquidation Strategies:
Auctioning: Selling off the portfolio to select clients or brokers.
Prop Book: Taking positions onto the TM's prop books at a fair price.
Market Closeout: Executing trades in the market to net off positions.

[Internal Control Guidelines for Clearing and Trading Members]
Source File: internal-controls-for-clearing-and-trading-members.pdf

Adequacy of Systems:
Members must ensure systems and connections operate properly and have adequate capacity.
Must have adequate trade execution, recording, reporting, clearing, and settlement procedures.
Must have sufficient staff with adequate knowledge and training.

Planning and Assessment:
Members should establish comprehensive planning and assessment programs to test system operation, capacity, and security.
Programs can be established under outsourcing arrangements.

Compliance:
Members must appoint one or more compliance officers competent to advise on the application of these guidelines and Rules.

 [NSE Clear Backtesting Policy]
Source File: nse-clear-backtesting-policy.pdf
Definition: Backtesting compares margin collected against a portfolio of positions to the realized losses at the end of the liquidation period had a default occurred.

Frequency: Backtests are carried out daily.
Parameters Tested:
Every single client portfolio.
Every single contract (underlying and expiry date combination).

Process:
Isolate client portfolio and margin collected.
Determine actual (realized) profit/loss using the Margin Period of Risk (MPOR).
Identify instances where margin was insufficient (exceptions).

Governance:
Carried out by NSE Derivatives Risk Management Team.
Results shared with NSE Clear Management team.
Exceptions reported to Derivatives Risk Management Committee and Derivatives Market Oversight Committee quarterly.


[NSE Derivatives Settlement Guarantee Fund Rules]
Source File: nse-derivatives-settlement-guarantee-fund-rules.pdf

Name: NSE Derivatives Settlement Guarantee Fund ("the Guarantee Fund").

Administration:
Trustees (Board of Directors of NSE or appointed persons) are the controlling body.
NSE acts as the secretary.

Assets:
Held in the name of the Fund or a nominee company.
Includes contributions from members, penalties, interest, and other sources.

Claims:
The Fund is used to discharge claims accepted by the trustees and obligations arising from the derivatives exchange business.
Used only after the defaulter's assets are exhausted.

[NSE Derivatives Investor Protection Fund Rules]
Source File: nse-derivatives-investor-protection-fund-rules.pdf
Purpose: To provide compensation to investors (Claimants) in the Derivatives Market.

Claimant Definition: Any client of a Trading Member and/or Non-Executing Member making a genuine and bonafide claim.

Claim Eligibility:
Claims relate to defalcation or fraudulent misuse of property/authority by a Trading Member.
Claimant must have given authority over property to the Member.


Trustees: Acquire, hold, and administer the Assets of the Fund.

Claims Process:
Claims must be submitted to Trustees.
Trustees refer claims to the Management Committee for consideration.
Payment is made from the Fund's assets (NSE/NSE Clear money is not available for these claims)


        [OFFICIAL_FAQ]
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

        Q:What is a derivative?
        A:A derivative is a financial instrument whose characteristics and value depend on an underlying asset. The asset can be an equity, currency, bond, interest rate, commodity or even the weather.

        Q:What are the different types of derivatives instruments?
        A:Derivatives can be broadly classified into four categories:
Forwards
Futures
Options
Swaps

        Q:What are the types of derivatives being traded on NEXT?
        A:NEXT currently offers equity futures contracts. These are classified into: Equity Index Futures and Single Stock Futures

        Q:What is a futures contract?
        A:Simply put, a futures contract is an agreement to exchange a pre-specified asset at a pre-specified price on a pre-specified date in the future.
The futures contract can be bought or sold on the NSE just like any other security.
Futures contracts require either physical delivery of the asset or settlement in cash.
All futures contracts on NEXT are settled in cash. This means that you realise your profit or loss in cash rather than delivering or receiving the physical asset.

        Q:What is the structure in the derivatives market?
        A:At the top of the hierarchy, is the Clearing House (NSE Clear), which is a wholly owned subsidiary of the NSE.
The Clearing House novates all transactions and is therefore the legal counterparty to both parties. The Clearing House essentially becomes the buyer to every seller and seller to every buyer and ensures that all parties fulfil their obligations.
Clearing Members are under the Clearing House. They are either banks or financial institutions that are responsible for clearing, settlement and risk monitoring of Trading Members.
Clients open accounts with Trading Members who then execute orders on their behalf i.e. buy and sell derivative contracts for them.

        Q:How does NEXT manage risks associated with these contracts?
        A:Investors require Initial Margin to enter into any futures position. This is a minimum (good faith) deposit required from an investor for the duration of an open contract. These margin requirements apply to both the buyer and seller.
Initial Margins for each contract are ordinarily reviewed and announced every quarter. For more information see Market Notices
Daily profits or losses on an investor’s position are either debited from or credited to the investor’s trading account.
If the investor’s trading account has insufficient funds to cover a loss, the investor will be required to add sufficient funds to the account or to close out (exit) their position.
The regular administration of margins prevents participants from accumulating large unpaid losses which could impact the financial positions of other market users (systemic risk).

        Q:What is Variation Margin?
        A:Unlike the cash market (equities market) where profits or losses re only realised when the instrument is sold, trading on NEXT means that investors receive or pay profits or losses on a daily basis.
This payment is known as the Variation Margin and is equal to the difference in the value of the investor’s position from day to day.

        Q:How does one invest on NEXT?
        A:NEXT contracts can be bought and sold through the Derivatives Trading Members of the NSE.
In order to open a trading account, the investor should be onboarded by an approved Trading Member. This involves the completion of various documents including member-client agreements, Know-Your-client (KYC) and risk disclosure documentation.
To begin trading, the investor must deposit cash and/or other collateral with the Trading Member as stipulated by the Trading Member.

        Q:Which brokers are approved derivatives Trading Members?
        A:An updated list of Trading Members is available on the Accredited Members page.

        Q:What are the benefits of trading derivatives on NEXT?
        A:Derivatives are more flexible than the underlying instruments while the value is still based on the price of the underlying assets.
Derivatives contracts can be leveraged. This means that you can trade a position that is larger than the amount of cash required upfront.
Derivatives can be used to protect against adverse price movements. You can therefore use derivatives to protect your existing portfolio of shares for example.
Derivatives are cheaper to trade compared to trading shares.

        Q:What is the difference between the derivatives market and the cash/spot market?
        A:In Spot Market, The securities have an infinite lifespan
Full cash or value of shares paid upfront.
For example, a trade worth KES 100,000 will require KES 100,000 upfront.
Profit/Loss is realised after exiting the position.

      In Derivatives Market, The contracts have an expiry date.
      A good faith deposit (margin) is paid upfront.
For example, a trade worth KES 100,000 will require approximately KES 10,000 upfront.
Profit/Loss realised on a daily basis.

        Q:How are these NEXT contracts closed out?
        A:All NEXT futures contracts have closeout dates on which the contract expires. The futures contracts are named according to their expiry month. For example, 16 DEC21 SCOM represents a Safaricom futures contract that will expire on 16th December 2021.
NEXT Equity Index Futures and NEXT Single Stock Futures expire on the third Thursday of the relevant expiry month. If that day is a holiday, then expiry is on the previous business day.
On the expiry date of a particular contract, the contract is terminated by the exchange and it ceases to exist. The exchange then refunds Initial Margins to investors.

        Q:How is Clearing and Settlement done on NEXT?
        A:The Clearing House (NSE Clear), through appointed Clearing Members computes the obligations of investors who have traded through Trading Members. These obligations detail the amounts an investor needs to pay or receive in terms of margins. The computation of these obligations is the “clearing” process. The actual flow of cash to satisfy the obligations is the “settlement” process. The clearing and settlement process is performed daily for all trades executed on NEXT.
Settlement is done on a T + 1 basis. This means that parties to a trade satisfy obligations arising from the transaction one day after the trade is executed. All futures contracts on NEXT are settled in cash on a daily basis.

        Q:What are the current futures contracts offered by the NSE?
        A:The NSE currently offers futures contracts based off an equity index and a select number of stocks listed at the Exchange
At the moment, the specific futures contracts are:
Index:
NSE 25 Share Index (N25I)
Mini NSE 25 Share Index (25MN)
Single Stock:
Safaricom Plc (SCOM);
KCB Group Plc (KCBG);
Equity Group Holdings Plc (EQTY);
ABSA Bank Kenya Plc (ABSA);
East African Breweries Ltd (EABL); and
British American Tobacco Kenya Plc (BATK).

        Q:What does 1 futures contract represent?
        A:Equity index futures
NSE 25 Share Index – 1 index point represents KES 100. Therefore, if the contract value changes by 1 point, you will gain or lose KES 100.
Mini NSE 25 Share Index – 1 index point represents KES 10. Therefore, if the contract value changes by 1 point, you will gain or lose KES 10.
Single stock futures
For stocks trading below KES 100, 1 contract represents 1,000 underlying shares. Therefore, if the contract value changes by KES 1, you will gain or lose KES 1,000.
For stocks trading above KES 100, 1 contract represents 100 underlying shares. Therefore, if the contract value changes by KES 1, you will gain or lose KES 100.

        Q:What is the selection criteria for single stock futures?
        A:In order to be eligible to trade as a future, the underlying stock:

Has to be listed on the NSE;
Has to be a constituent of the NSE 25 share index;
Must have traded an average daily turnover of KES 7,000,000 for six months prior to review; and
Must have a market capitalization of at least KES 50 Billion

        Q:How often does the NSE review stocks to determine whether they merit to be listed as futures contracts?
        A:The NSE undertakes quarterly reviews of stocks to determine whether to list or delist them as futures contracts.

        Q:Does the 10% price limit rule apply to futures contracts?
        A:Yes. The rule applies to futures contracts in the same way it applies to the equities market. Futures prices may move a maximum of 10% during the trading day unless the NSE communicates otherwise.

        Q:Does an investor have to hold a futures contract until maturity of the contract?
        A:No. An investor can trade and offload their futures contract at any point in time during the duration of the contract. This could even be done on an intra-day basis where the investor buys and sells contracts on the same day.

        Q:Can one benefit from futures contracts if the market is going down?
        A:Yes. The derivatives market offers dual opportunities for investors to profit when the market goes up (by initiating a buy/long position) or when the market goes down (by initiating a sell/short position).

        Q:Is there an online trading platform?
        A:Yes. The NSE has provided an online trading platform for the derivatives market. Investors may request view access from their Trading Members. The Trading Member may also grant trading rights to the investor at their own discretion.

        Q:Will I receive dividends from futures contracts?
        A:No. Futures contracts do not represent ownership of the underlying asset therefore investors do not receive dividends or voting rights by virtue of holding single stock futures. Dividends are accounted for in the daily valuation of the futures contracts.

        Q:What is the minimum amount of capital I need to start trading?
        A:This depends on the futures contract you would like to trade.
For example, to trade one ABSA futures contract you will require a minimum of approximately KES 1,500 while a single Safaricom contract will require approximately KES 4,600 and one Mini NSE 25 Share index contract requires approximately KES 4,800.
See Market Notices for an updated list of margin requirements.

        Q:What time is the market open?
        A:The futures market is open for trading from 9:00 am to 3:00 pm on Monday to Friday except public holidays.

        Q:What are options?
        A:Options are financial derivatives that provide the buyer of the option (holder), the right but not the obligation, to buy or sell an underlying asset at a predetermined price within a specified period. There are two primary types:

Call Options: an option to buy an asset (the underlying) for a specified price (the strike or exercise price), on or before a specified date.
Put Options: option is an option to sell an asset for a specified price on or before a specified date. Remember this by thinking that the buyer can put the asset on to someone else (the seller of the option), demanding the pre-agreed sum in exchange.

        Q:What is the strike/exercise price?
        A:This is the predetermined price at which the underlying asset will be bought or sold.

        Q:Who is the holder of an option?
        A:The holder has the right but not the obligation to exercise the option to buy or sell the underlying security. The holder pays a premium to the writer of the option.

        Q:Who is the writer of an option?
        A:The writer receives a premium from the holder and is obligated to sell or buy the underlying if the holder exercises the option.

        Q:How do options differ from futures contracts?
        A:While both are derivatives, options give the right without obligation to buy or sell the underlying asset, whereas futures contracts obligate both parties to transact at a set price on a future date.

        Q:What underlying assets are available for options trading on the NSE?
        A:The NSE offers options on the existing futures contracts:

Single stock futures on select listed companies.
Index futures based on NSE indices.

        Q:Who can participate in options trading on the NSE?
        A:Both individual and institutional investors

        Q:What are the benefits of trading options?
        A:Leverage: Control a large position with a relatively small investment.
Risk Management: option holders are protected from adverse movements in the market whereby risk is limited to the premium paid while potential gains are unlimited. Investors can incorporate options trading strategies to their portfolio’s to enhance returns and manage market risk.
Flexibility: Implement various strategies to profit in different market conditions.

        Q:How does options settlement work on the NSE?
        A:The NSE Options contracts are settled through cash settlement:

        Q:What risks are associated with options trading?
        A:Writers of the options are exposed to capital losses if holders exercise their options. This could potentially be more than the amount invested.

        Q:How can I start trading options on the NSE?
        A:Open an account with a derivatives licensed trading member (stockbroker or investment bank). The member will provide you with access to the online trading platform.
        















        
        """

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def get_embeddings_batch(self, texts):
        if not texts: return []
        sanitized = [t.replace("\n", " ") for t in texts]
        res = self.client.embeddings.create(input=sanitized, model=EMBEDDING_MODEL)
        return [d.embedding for d in res.data]

    def get_embedding(self, text):
        return self.get_embeddings_batch([text])[0]

    def build_knowledge_base(self):
        seeds = [
           "https://www.nse.co.ke/",
            "https://www.nse.co.ke/site-map/",
            "https://www.nse.co.ke/home/",
            "https://www.nse.co.ke/about-nse/",
            "https://www.nse.co.ke/our-story/",
            "https://www.nse.co.ke/contact-us/",
            "https://www.nse.co.ke/faqs/",
            "https://www.nse.co.ke/leadership/",
            "https://www.nse.co.ke/listed-companies/",
            "https://www.nse.co.ke/corporate-bonds/",
            "https://www.nse.co.ke/government-bonds/",
            "https://www.nse.co.ke/equities-market/",
            "https://www.nse.co.ke/exchange-traded-funds/",
            "https://www.nse.co.ke/share-price/",
            "https://www.nse.co.ke/market-statistics/",
            "https://www.nse.co.ke/data/",
            "https://www.nse.co.ke/derivatives/equity-index-futures/",
            "https://www.nse.co.ke/derivatives/accredited-members/",
            "https://www.nse.co.ke/derivatives/",
            "https://www.nse.co.ke/derivatives/about-next",
            "https://www.nse.co.ke/derivatives/single-stock-futures/",
            "https://www.nse.co.ke/derivatives/options-on-futures/"
            "https://www.nse.co.ke/derivatives/faq/",
            "https://www.nse.co.ke/growth-enterprise-market-segment/",
            "https://www.nse.co.ke/green-bonds/",
            "https://www.nse.co.ke/training/",
            "https://www.nse.co.ke/usp/",
            "https://www.nse.co.ke/ibuka-2/",
            "https://www.nse.co.ke/ibuka/",
            "https://www.nse.co.ke/m-akiba/",
            "https://www.nse.co.ke/list-of-trading-participants/",
            "https://www.nse.co.ke/real-estate-investment-trusts/",
            "https://www.nse.co.ke/main-investment-market-segment/",
            "https://www.nse.co.ke/alternative-investment-market-segment/",
            "https://www.nse.co.ke/sustainability/",
            "https://www.nse.co.ke/nominated-advisors/",
            "https://www.nse.co.ke/corporate-actions/",
            "https://www.nse.co.ke/listed-company-announcements/",
            "https://www.nse.co.ke/privacy-policy/",
            "https://www.nse.co.ke/trading-participant-financials/",
            "https://www.nse.co.ke/csr/",


            #come back to append pdfs 
            "https://www.nse.co.ke/derivatives/next-education-resources/",
            "https://www.nse.co.ke/policy-guidance-notes/",
            #Data Services Links
            "https://www.nse.co.ke/dataservices/",
            "https://www.nse.co.ke/dataservices/market-statistics/",
            "https://www.nse.co.ke/dataservices/market-data-overview/",
            "https://www.nse.co.ke/dataservices/real-time-data/",
            "https://www.nse.co.ke/dataservices/end-of-day-data/",
            "https://www.nse.co.ke/dataservices/historical-data/",
            "https://www.nse.co.ke/dataservices/historical-data-request-form/",
            "https://www.nse.co.ke/dataservices/international-securities-identification-number-isin/"
        ]
        
        hardcoded_pdfs = [
             "https://www.nse.co.ke/wp-content/uploads/nse-equities-trading-rules.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-listing-rules.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Groundrules-nse-NSE-BankingSector-share-index_-V1.0.pdf",
            "https://www.nse.co.ke/wp-content/uploads/GroundRules-NSE-Bond-Index-NSE-BI-v2-Index-final-2.pdf",
            "https://www.nse.co.ke/wp-content/uploads/groundrules-nse-20-share-index_-v1.6.pdf",
            "https://www.nse.co.ke/wp-content/uploads/GroundRules-NSE-10v2-Share-Index.pdf",
            "https://www.nse.co.ke/wp-content/uploads/groundrules-nse-25-share-index_-v1.4.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-market-participants-rules.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Equity-Trading-Rules-Amended-Jul-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-fixed-income-trading-rules.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Fixed-Income-Trading-Rules-2024.pdf",
            "https://www.nse.co.ke/wp-content/uploads/nse-derivatives-rules.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/nse-derivatives-investor-protection-fund-rules.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/nse-derivatives-rules-1-1.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/nse-derivatives-settlement-guarantee-fund-rules.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/default-handling-procedure.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/initial-margin-calculation-methodology.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/mark-to-market-methodology-april-2021.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/nse-clear-backtesting-policy.pdf",
            "https://www.nse.co.ke/wp-content/uploads/policy-guidance-note-for-exchange-traded-funds.pdf",
            "https://www.nse.co.ke/wp-content/uploads/policy-guidance-note-for-green-bonds.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-ESG-Disclosures-Guidance-Manual.pdf",
            "https://www.nse.co.ke/wp-content/uploads/BBO-standards.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/nse-derivatives-membership-criteria-and-fees-february-2021.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/corporate-action-management-guide.pdf",
            "https://www.nse.co.ke/wp-content/uploads/HOW-TO-BECOME-A-TRADING-PARTICIPANT-.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-OPERATIONAL-GUIDELINES-FOR-THE-BOND-QUOTATIONS-BOARD.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Day-Trading-Operational-Guidelines.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-direct-market-access-october-2019.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-financial-resource-requirements-for-market-intermediaries.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-managementsupervision-and-internal-control-of-cma-licensed-entities-may-2012.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-the-prevention-of-money-laundering-and-terrorism-financing-in-the-capital-markets.pdf",
            "https://www.nse.co.ke/wp-content/uploads/East-African-Exchanges-EAE-20-Share-Index-Methodology-F-.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Implied-Yields-Yield-Curve-Generation-Methodology-1.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2024/08/PL-Calculator-User-Manual.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/internal-controls-for-clearing-and-trading-members.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/internal-controls-for-clearing-and-trading-members.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2025/03/Mini-NSE-10-Index-Futures-Product-Report.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/nse-clear-status-on-pfmi-principles-april-2021.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2025/03/Product-Report-Options-on-Futures-August-2024-Approved.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-2025-2029-Strategy.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-ESOP-Public-Announcement-2025.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE_Press-Release-Nairobi-Securities-Exchange-Plc-takes-a-bold-step-to-expand-investment-access-for-retail-investors-1.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Press-Release-Nairobi-Securities-Exchange-Plc-Appoints-Sterling-Capital-Limited-as-a-market-maker-in-the-NEXT-Derivatives-Market.pdf",
            "https://www.nse.co.ke/wp-content/uploads/Press-Release-Nairobi-Securities-Exchange-Plc-Launches-Banking-Sector-Index.pdf",
            "https://www.nse.co.ke/wp-content/uploads/broker-back-office-prequalified-vendors.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/derivatives-document-1.pdf"

        ]

        print("🕷️ Crawling NSE website...")
        found_pages, found_pdfs = self.crawl_site(seeds)
        all_urls = list(set(found_pages + found_pdfs + hardcoded_pdfs))
        
        print(f"📝 Found {len(all_urls)} total documents.")
        total_chunks = self.scrape_and_upload(all_urls)
        
        return f"Knowledge Base Updated: {total_chunks} chunks uploaded to Pinecone.", []

    def scrape_and_upload(self, urls):
        total_uploaded = 0
        
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
                        "text": chunk[:30000], 
                        "source": url,
                        "date": datetime.date.today().isoformat(),
                        "type": ctype
                    }
                    vectors.append({"id": vector_id, "values": embeddings[i], "metadata": metadata})
                
                return vectors
            except Exception as e:
                print(f"Error processing {url}: {e}")
                return None

        vectors_to_upload = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_url, u): u for u in urls}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res: vectors_to_upload.extend(res)
        
        for i in range(0, len(vectors_to_upload), 100):
            batch = vectors_to_upload[i:i+100]
            try:
                self.index.upsert(vectors=batch)
                total_uploaded += len(batch)
            except Exception as e:
                print(f"Pinecone Upsert Error: {e}")
            time.sleep(0.2)
            
        return total_uploaded

    def generate_context_queries(self, query):
        today = datetime.date.today().strftime("%Y-%m-%d")
        prompt = f"Generate 3 search queries for: '{query}'\nDate: {today}\n1. Keyword\n2. Concept\n3. Doc type\nOutput 3 lines."
        try:
            res = self.client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
            return [q.strip() for q in res.choices[0].message.content.split('\n') if q.strip()]
        except: return [query]

    def answer_question(self, query):
        context_text = self.get_static_facts() + "\n\n"
        visible_sources = set()
        
        try:
            queries = self.generate_context_queries(query)
            q_emb = self.get_embedding(queries[0])
            
            results = self.index.query(vector=q_emb, top_k=15, include_metadata=True)
            
            if results['matches']:
                docs = [m['metadata']['text'] for m in results['matches']]
                metas = [m['metadata'] for m in results['matches']]
                
                tokenized_query = query.lower().split()
                tokenized_docs = [doc.lower().split() for doc in docs]
                
                bm25 = BM25Okapi(tokenized_docs)
                doc_scores = bm25.get_scores(tokenized_query)
                
                final_ranking = []
                for i, match in enumerate(results['matches']):
                    hybrid_score = match['score'] + (doc_scores[i] * 0.1)
                    
                    if "[OFFICIAL_FAQ]" in docs[i]: hybrid_score += 0.5
                    if "[OFFICIAL_FACT_SHEET]" in docs[i]: hybrid_score += 1.0
                    
                    final_ranking.append((hybrid_score, docs[i], metas[i]['source']))
                
                final_ranking.sort(key=lambda x: x[0], reverse=True)
                
                for _, text, source in final_ranking[:5]:
                    context_text += f"\n[Source: {source}]\n{text}\n---"
                    visible_sources.add(source)

        except Exception as e:
            print(f"Retrieval Error: {e}")
        
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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _fetch_url(self, url):
        headers = {'User-Agent': 'Mozilla/5.0'}
        return self.session.get(url, headers=headers, verify=False, timeout=10)

    def crawl_site(self, seed_urls):
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
        tag = "[GENERAL]"
        if "statistics" in url: tag = "[MARKET_DATA]"
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