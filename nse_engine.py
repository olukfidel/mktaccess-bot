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
        -Pre-Open Session: 09:00 am - 09:30 am

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

[OFFICIAL_NSE_LISTED_COMPANIES_NOVEMBER_2025]
SOURCE: Nairobi Securities Exchange – Full List of Listed Companies (as at 26 November 2025)

MAIN INVESTMENT MARKET SEGMENT (MIMS) – EQUITY
1. ABSA Bank Kenya Plc – ABSA
2. Bakri Energy Kenya Ltd – BAKRI
3. B.O.C Kenya Plc – BOC
4. British American Tobacco Kenya Plc – BAT
5. Carbacid Investments Plc – CARB
6. Centum Investment Company Plc – CTUM
7. Diamond Trust Bank Kenya Ltd – DTK
8. East African Breweries Plc – EABL
9. Equity Group Holdings Plc – EQTY
10. HF Group Plc – HFCK
11. I&M Group Plc – IMH
12. Jubilee Holdings Ltd – JUB
13. KCB Group Plc – KCB
14. KenGen Company Plc – KEGN
15. Kenya Power & Lighting Company Plc – KPLC
16. Kenya Reinsurance Corporation Ltd – KNRE
17. Liberty Kenya Holdings Plc – LBTY
18. Nairobi Securities Exchange Plc – NSE
19. NCBA Group Plc – NCBA
20. Safaricom Plc – SCOM
21. Standard Chartered Bank Kenya Ltd – SCBK
22. Stanbic Holdings Plc – SBIC
23. The Co-operative Bank of Kenya Ltd – COOP
24. TotalEnergies Marketing Kenya Plc – TOTL
25. Umeme Limited – UMME
26. BK Group Plc (Bank of Kigali) – BKG
27. Britam Holdings Plc – BRIT
28. CIC Insurance Group Ltd – CIC
29. Flame Tree Group Holdings Ltd – FTGH
30. Home Afrika Ltd – HAFR
31. ILAM Fahari I-REIT – ILAFM
32. Kakuzi Plc – KUKZ
33. Kapchorua Tea Kenya Plc – KAPC
34. Limuru Tea Plc – LIMT
35. Longhorn Publishers Plc – LKL
36. Nation Media Group Plc – NMG
37. NewGold Issuer Ltd – NEWGOLD
38. Olympia Capital Holdings Ltd – OCH
39. Sameer Africa Plc – SMER
40. Sanlam Kenya Plc – SLAM
41. Sasini Plc – SASN
42. TPS Eastern Africa (Serena) Ltd – TPSE
43. Trans-Century Plc – TCL
44. WPP Scangroup Plc – SCAN
45. Williamson Tea Kenya Plc – WTJK

GROWTH ENTERPRISE MARKET SEGMENT (GEMS)
1. Homeboyz Entertainment Plc – HBE

FIXED INCOME SECURITIES (CORPORATE BONDS & COMMERCIAL PAPER)
Over 120 active fixed-income instruments from Government of Kenya, Kenya Electricity Generating Company, Centum Investment, Britam, Family Bank, Housing Finance, I&M Bank, KCB Bank, NCBA Bank, Co-operative Bank, ABSA Bank, Stanbic Bank, etc. Full list available at https://www.nse.co.ke/bonds/

REAL ESTATE INVESTMENT TRUSTS (REITs)
1. ILAM Fahari I-REIT (I-REIT) – ILAFM
2. Laptrust Imara I-REIT (I-REIT) – Under registration
3. Acorn D-REIT & I-REIT – In process

EXCHANGE TRADED FUNDS (ETFs)
1. NewGold ETF – Tracks price of physical gold

TOTAL LISTED EQUITY SECURITIES: 66 companies (65 MIMS + 1 GEMS)
TOTAL MARKET CAPITALIZATION (Nov 2025): Approximately KES 2.4 trillion
CEO: Mr. Frank Mwiti
Trading Hours: Monday–Friday, 10:00 a.m. – 3:00 p.m. (Pre-open 9:30–10:00 a.m.)
Official Website: https://www.nse.co.ke
Data Room: https://www.nse.co.ke/market-data/


[OFFICIAL_NSE_10_SHARE_INDEX_FUTURES_RULES_V2]
SOURCE: https://www.nse.co.ke/derivatives/equity-index-futures/ – NSE 10 Share Index Futures Ground Rules v2 (November 26, 2025)

NEXT Equity Index Futures are derivative instruments that give investors exposure to price movements on an underlying index. Market participants can profit from the price movements of a basket of equities without trading the individual constituents.

An index futures contract gives investors the ability to buy or sell an underlying listed financial instrument at a fixed price on a future date. These products are cash settled and easily accessible via NEXT members. The NSE shall initially construct Equity Index Futures contracts based on the NSE 10 Share Index.

NSE Derivatives Market

Category of contract: Equity Index Future

Underlying financial instrument: Equity Index listed on the NSE E.g. NSE10 Share Index – N10I

System code: Jun19 N10I

Contract months: Quarterly (March, June, September and December).

Expiry dates: Third Thursday of expiry month. (If the expiry date is a public holiday then the previous business day will be used.)

Expiry times: 15H00 Kenyan time.

Listing program: Quarterly

Valuation method on expiry: Based on the volume weighted average price (VWAP) of the underlying instrument for liquid contracts, and the theoretical price (spot + cost of carry) for illiquid contracts.

Settlement methodology: Cash settlement.

Contract size: One index point equals one hundred Kenyan Shillings. (KES 100.00)

Minimum price movement (Quote spread): One index point (KES 100.00)

Initial Margin requirements: As determined by the NSE Methodology.

Mark-to-market: Explicit daily. Based on the volume weighted average price (VWAP) of the underlying for liquid contracts, and the theoretical price (spot + cost of carry) for illiquid contracts.

Market trading times: As determined by the NSE 09H30 to 15H00 Kenyan time

Market fees:

| Participant | Percentage |
|-------------|------------|
| NSE Clear   | 0.02%      |
| Clearing Member | 0.02% |
| Trading Member | 0.08% |
| IPF Levy    | 0.01%      |
| CMA Fee     | 0.01%      |
| TOTAL       | 0.14%      |

The percentages indicated above will be used to calculate the fees based on the notional contract value.

NEXT Equity Index Futures allow investors to get some form of "insurance" for their stock portfolio by protecting portfolios from potential price declines;

Benefits of trading NEXT Equity Index Futures:

- Price transparency and liquidity. These contracts can be sold as easily as they can be bought;

- Lower transaction fees than those incurred when buying or selling the basket of securities making up the index;

- Reduction of counter-party risk a result of trading via the exchange; and

- Centralized clearing.



[OFFICIAL_NSE_SINGLE_STOCK_FUTURES_RULES_V2]
SOURCE: https://www.nse.co.ke/derivatives/single-stock-futures/ – NEXT Single Stock Futures Ground Rules v2 (November 26, 2025)

NEXT Single Stock Futures are derivative instruments that give investors exposure to price movements on an underlying stock. Parties agree to exchange a specified number of stocks in a company for a price agreed today (the futures price). NEXT Single Stock Futures will initially be cash settled.

Benefits:
- Provide an effective and transparent hedge against unfavorable share price movements;
- They are liquid and easy to trade instruments;
- Positions in single stock futures allow investors to benefit from downwards or upwards movement of share prices; and
- Investors can have exposure on share price movements without owning the underlying share.

Contract Specifications:
- Category of contract: Single Stock Future
- Underlying financial instrument: Single stock listed on the NSE E.g. Equity Group Holdings Plc. – EQTY
- System code: Jun19 EQTY
- Contract months: Quarterly (March, June, September and December).
- Expiry dates: Third Thursday of expiry month. (If the expiry date is a public holiday then the previous business day will be used.)
- Expiry times: 15H00 Kenyan time.
- Listing program: Quarterly
- Valuation method on expiry: Based on the volume weighted average price (VWAP) of the underlying instrument for liquid contracts, and the theoretical price (spot + cost of carry) for illiquid contracts.
- Settlement methodology: Cash settlement.
- Contract size: For shares trading below KES 100: One contract equals 1,000 underlying shares. For shares trading above KES 100: One contract equals 100 underlying shares.
- Minimum price movement (Quote spread):

| Price Range   | Tick Size (KES) |
|---------------|-----------------|
| Below 100.00  | 0.01            |
| ≥ 100.00 < 500.00 | 0.05        |
| ≥ 500.00      | 0.25            |

- Initial Margin requirements: As determined by the NSE Methodology.
- Mark-to-market: Explicit daily. Based on the volume weighted average price (VWAP) of the underlying for liquid contracts, and the theoretical price (spot + cost of carry) for illiquid contracts.

Trading Hours: Market trading times: As determined by the NSE 09H30 to 15H00 Kenyan time.

Market fees:

| Participant   | Percentage |
|---------------|------------|
| NSE Clear     | 0.025%     |
| Clearing Member | 0.025%  |
| Trading Member | 0.10%   |
| IPF Levy      | 0.01%     |
| CMA Fee       | 0.01%     |
| TOTAL         | 0.17%     |

The percentages indicated above will be used to calculate the fees based on the notional contract value.


[OFFICIAL_NSE_OPTIONS_ON_FUTURES_RULES_V1]
SOURCE: https://www.nse.co.ke/derivatives/options-on-futures/ – NSE Options on Futures Ground Rules v1 (November 26, 2025)

An option is a contract that gives the buyer the right, but not the obligation, to sell or buy a particular asset at a particular price, on or before a specified date. The seller of the option, conversely, assumes an obligation in respect of the underlying asset upon which the option has been traded.

Types:
- A call option is an option to buy an asset (the underlying) for a specified price (the strike or exercise price), on or before a specified date.
- A put option is an option to sell an asset for a specified price on or before a specified date.

Buyer and Seller: The buyer of an options contract is said to be long, or the holder or owner of the contract. The seller of an options contract is said to be short, or the writer of the contract.

Underlying Assets: Options are available on a variety of underlying assets – physical assets, like oil and sugar, and financial assets, such as cash shares and FX forwards. The option may be based on a futures contract, where the underlying asset is a future; these are known as options on futures.

NSE Derivatives Market - Options on Single Stock Future

Category of Contract: Options on Single Stock Future

Underlying Financial Instrument: Single stock futures listed on the NSE – Safaricom Plc – SCOM

System Code: 19 SEP 24 SCOM 20.00 CALL/PUT

Contract Months: Monthly or quarterly (March, June, September and December).

Expiry Dates: The third Thursday of every expiry month. (If the expiry date is a public holiday then the previous business day will be used.)

Expiry Times: At 15H00 Kenyan time.

Listing Program: Monthly or Quarterly

Valuation Method on Expiry: This will be based on the volume weighted average price of the underlying for liquid contracts, and the theoretical price (spot + cost of carry) for illiquid contracts.

Settlement Methodology: Cash settled through the NSE.

Contract Size: One options contract equals 1 underlying single stock futures contract.

Minimum Price Movement (Quote Spread): In Kenyan Shillings per two decimal places. (KES 0.01)

Mark-to-Market: Explicit daily. This is based on the volume weighted average price of the underlying for liquid contracts, and the theoretical price (spot + cost of carry) for illiquid contracts.

Market Trading Times: As determined by the NSE (09H00 to 15H00) local Kenyan time.

Market Fees:

| Participant    | Percentage |
|----------------|------------|
| NSE Clear      | 0.0125%    |
| Clearing Member| 0.0125%    |
| Trading Member | 0.05%      |
| IPF Levy       | 0.005%     |
| CMA Fee        | 0.005%     |
| TOTAL          | 0.085%     |

The percentages indicated above will be used to calculate the fees based on the notional contract value.

Benefits of Trading NEXT Options on Futures:
- Leverage: Control a large position with a relatively small investment.
- Risk Management: option holders are protected from adverse movements in the market whereby risk is limited to the premium paid while potential gains are
- Investors can incorporate options trading strategies to their portfolio’s to enhance returns and manage market risk.
- Flexibility: Implement various strategies to profit in different market conditions.


[OFFICIAL_NSE_DERIVATIVES_MEMBERSHIP_REQUIREMENTS_2025]
SOURCE: https://www.nse.co.ke/derivatives/membership-requirements/ – NSE NEXT Derivatives Membership Requirements (as at November 26, 2025)

Undertaking to Comply with NSE Rules: Signed undertaking to comply with Rule 3 on NEXT Membership.

Key Responsibilities:
- Clearing Member: Perform clearing and settlement for the market.
- Trading Member: Trade on behalf of clients or own proprietary account.
- Trading Members (Proprietary Trading): Trade only for own proprietary account.
- Non-Executing Member (Custodians): Acceptance of trades executed on behalf of custodial clients in order to facilitate settlement.

Net Worth/Capital Adequacy:
- Clearing Member: KES 1 Billion and as mandated by the Central Bank of Kenya.
- Trading Member: Thirteen weeks operating costs.
- Trading Members (Proprietary Trading): Ten weeks operating costs.
- Non-Executing Member (Custodians): Thirteen weeks operating costs.

NEXT Membership Fees:
- Clearing Member: Joining: KES 500,000; Annual: KES 100,000.
- Trading Member: Joining: KES 100,000; Annual: KES 100,000.
- Trading Members (Proprietary Trading): Joining: KES 50,000; Annual: KES 50,000.
- Non-Executing Member (Custodians): Joining: KES 100,000; Annual: KES 100,000.

Guarantee Fund Deposit: All clearing members must make an undertaking with the clearing house to cover their obligations to the Clearing House to the extent of their market positions and their balance sheets.

Investor Protection Fund:
- Clearing Member: N/A.
- Trading Member: Onetime, refundable maximum of KES 200,000. Contributions also based on regular directives issued by the NSE.
- Trading Members (Proprietary Trading): Onetime, refundable maximum of KES 100,000. Contributions also based on regular directives issued by the NSE.
- Non-Executing Member (Custodians): Onetime, refundable maximum of KES 200,000. Contributions also based on regular directives issued by the NSE.

| Requirement                  | Clearing Member                                                                 | Trading Member                                                                 | Trading Members (Proprietary Trading)                                          | Non-Executing Member (Custodians)                                               |
|------------------------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Undertaking to Comply with NSE Rules | Signed undertaking to comply with Rule 3 on NEXT Membership.                     | Signed undertaking to comply with Rule 3 on NEXT Membership.                     | Signed undertaking to comply with Rule 3 on NEXT Membership.                     | Signed undertaking to comply with Rule 3 on NEXT Membership.                     |
| Key Responsibilities         | Perform clearing and settlement for the market.                                  | Trade on behalf of clients or own proprietary account.                          | Trade only for own proprietary account.                                         | Acceptance of trades executed on behalf of custodial clients in order to facilitate settlement. |
| Net Worth/Capital Adequacy   | KES 1 Billion and as mandated by the Central Bank of Kenya.                      | Thirteen weeks operating costs.                                                 | Ten weeks operating costs.                                                      | Thirteen weeks operating costs.                                                 |
| NEXT Membership Fees         | Joining: KES 500,000 Annual: KES 100,000                                         | Joining: KES 100,000 Annual: KES 100,000                                        | Joining: KES 50,000 Annual: KES 50,000                                          | Joining: KES 100,000 Annual: KES 100,000                                        |
| Guarantee Fund Deposit       | All clearing members must make an undertaking with the clearing house to cover their obligations to the Clearing House to the extent of their market positions and their balance sheets. | –                                                                              | –                                                                              | –                                                                              |
| Investor Protection Fund     | N/A                                                                              | Onetime, refundable maximum of KES 200,000. Contributions also based on regular directives issued by the NSE. | Onetime, refundable maximum of KES 100,000. Contributions also based on regular directives issued by the NSE. | Onetime, refundable maximum of KES 200,000. Contributions also based on regular directives issued by the NSE. |



[OFFICIAL_NSE_DERIVATIVES_ACCREDITED_MEMBERS_2025]
SOURCE: https://www.nse.co.ke/derivatives/accredited-members/ – NSE Derivatives (NEXT) Market Accredited Members (as at November 26, 2025)

Accredited Members (Trading Members):
| Member Name              | Address                                                                 | Contact Details                                                                 | Notes |
|--------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------|
| AIB-AXYS Africa          | The Promenade 5th Floor, General Mathenge Drive, Westlands, P.O. Box 43676-00100 | +254-020-7602525 / 0202226440                                                   | -     |
| Faida Investment Bank Ltd| Crawford Business park, Ground Floor, State House Road, P. O. Box 45236-00100 | +254-20-7606026-35                                                              | -     |
| Sterling Capital Ltd     | Delta Corner Annex, 5th Floor, Ring Road, Westlands, Nairobi, P.O. Box 45080-00100 | 2213914 / 244077 / 0723153219 / 0734219146                                      | -     |
| Standard Investment Bank Ltd | ICEA Building, 16th floor, P. O. Box 13714-00800                          | 2228963 / 2228967 / 2228969                                                     | -     |
| Genghis Capital Ltd      | 1st Floor, Purshottam Place Building, Westlands Road, P.O Box 9959-00100, Nairobi Kenya | +254 730145000 / +254 709185000                                                 | -     |
| NCBA Investment Bank     | Mara Rd. Upper-hill, P.O Box 44599-00100, Nairobi                        | +254 20 2884444 / +254 711 056444 / +254 732 156444                             | https://investment-bank.ncbagroup.com/ |
| Scope Markets            | Westide Towers, 4th Floor, Office 402 and 403, Lower Kabete Road, Westlands | +254 20 5005100, +254 730 831000, support@scopemarkets.co.ke                    | www.scopemarkets.co.ke |
| Kingdom Securities Limited | Co-operative Bank House, 5th Floor, Haile Selassie Avenue, Nairobi, P.O. Box 48231-00100 | +254 711 049039 / +254 20 2776000 / +254 703 027000, info@kingdomsecurities.co.ke | www.kingdomsecurities.co.ke |

Non-Executing Members / Custodians:
| Member Name              | Address                                                                 | Contact Details                                                                 | Website                  |
|--------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------|--------------------------|
| Co-operative Bank of Kenya | Co-operatives Bank House, Haile Selassie Avenue, P.O. Box 48231-0100, Nairobi, Kenya | +254 20 277 6000 / +254 703 027 000                                             | www.co-opbank.co.ke      |
| NCBA Custodial Services  | NCBA Bank Headquarters, Mara Rd, Upperhill, P.O. Box 44599-00100, Nairobi, Kenya | -                                                                               | www.ncbagroup.com        |

Clearing Members:
| Member Name              | Address                                                                 | Contact Details                                                                 | Website                  |
|--------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------|--------------------------|
| Co-operative Bank of Kenya | Co-operative Bank House, Haile Selassie Avenue, P.O. Box 48231-00100, Nairobi, Kenya | +254 20 277 6000 / +254 703 027 000                                             | www.co-opbank.co.ke      |

Notes: These are the accredited members for the NSE Derivatives (NEXT) Market. Categories include Trading Members, Non-Executing Members/Custodians, and Clearing Members. For more details on membership requirements, refer to NSE Derivatives Membership Requirements.


[OFFICIAL_NSE_TRADING_PARTICIPANTS_LIST_NOVEMBER_2025]
SOURCE: https://www.nse.co.ke/list-of-trading-participants/ – Full List of NSE Trading Participants (as at November 26, 2025)

### Full Trading Participants

| Name | Address | Phone | Email | Notes |
|------|---------|-------|-------|-------|
| Dyer & Blair Investment Bank Ltd - B02 | Goodman Tower, 7th floor, P.O. Box 45396 00100 | 0709930000 |  |  |
| Francis Drummond & Company Limited - B01 | Finance House, 14th Floor, Loita Street (Opposite GPO Kenyatta Avenue, Nairobi) P.O. Box 45465 00100 | 318690/318689 |  |  |
| Suntra Investment Bank Ltd - B07 | Nation Centre,7th Floor, P.O. Box 74016-00200 | 2870000 / 247530 / 2223330 / 2211846 |  |  |
| OMNI MARCHE SECURITE (OMS) AFRICA LTD - B08 | 4th Avenue Towers, 13th Floor, 4th Ngong Avenue, Upperhill P.O. Box 2151–00202, Kenyatta, Nairobi | 0709 004 330 / 0709 004 331 / 0709 004 332 / 0724 226 600 / 0709 004 300 | info@omsafrica.co.ke |  |
| SBG Securities Ltd - B09 | CfC Stanbic Centre, 58 Westlands Road, P. O. Box 47198 – 00100 | 3638900 |  |  |
| Kingdom Securities Ltd - B11 | Co-operative Bank House,5th Floor, P.O Box 48231 00100 | +254711049039, +254711049195, +254711049956, +254711049657 |  |  |
| AIB-AXYS Africa -B12 | The Promenade 5th Floor, General Mathenge Drive, Westlands, P.O. Box 43676- 00100 | +254-020-7602525 / 0202226440 |  |  |
| ABC Capital Ltd - B14 | Mezzanine, ABC Bank House, Woodvale Grove, Westlands P. O. BOX 34137 GPO O0100 Nairobi | (+254 20) 2246036, 2242534, 316143, 2241142, 2241046, 2241148 |  |  |
| Sterling Capital Ltd - B15 | Delta Corner Annex building – 5th Floor, Ring Road, P.O. Box 45080- 00100 | 2213914 / 244077 / 0723153219 / 0734219146 |  |  |
| Pergamon Investment Bank - B16 | 4th Floor, Delta Chambers, Waiyaki Way, Nairobi P.O Box 25749, Lavington | +254 709227100 |  |  |
| Faida Investment Bank Ltd - B17 | Crawford Business park, Ground Floor, State House Road, P. O. Box 45236-00100 | +254-20-7606026-35 |  |  |
| Standard Investment Bank Ltd - B20 | 16th floor JKUAT Building, Kenyatta Ave, Nairobi, Kenya | +254(0) 20 2277 000 / +254(0) 777 333 000 / +254(0) 793 333 000 |  |  |
| Kestrel Capital (EA) Limited - B21 | 2nd Floor, Orbit Place, Westlands Road, P.O. Box 40005-00100 | 251758 / 2251893 / 2251815 |  |  |
| Renaissance Capital (Kenya) Ltd -B24 | 10th Floor, Pramukh Tower, Westlands Road, Westlands, Nairobi Kenya | +254 (20) 368-2000 |  |  |
| Genghis Capital Ltd - B19 | 1st Floor, Purshottam Place Building, Westlands Road, P.O Box 9959-00100, Nairobi Kenya | +254 730145000 / +254 709185000 |  |  |
| NCBA Investment Bank Limited - B18 | 3rd Floor, NCBA Annex, Hospital Road, Upper Hill, P.O Box 44599-00100, Nairobi | +254 20 2884444, +254711056444, +254 732 156444 |  |  |
| Equity Investment Bank Limited - B26 | Equity Centre, Hospital Road, Upper Hill, P.O Box 75104 – 00200 | +254-20-2262477, +254-732-112477 |  |  |
| KCB Investment Bank | Kencom House 2nd Floor, P.O Box 48400 – 00100 | +254 711 012 000 / 734 108 200, +254 20 3270000 |  |  |
| Absa Securities Limited - B28 | Absa Headquarters, Waiyaki Way, PO Box 30120, 00100 | +254(732)130120/ +254(722)130120 |  |  |
| Capital A Investment Bank - B29 | Mayfair Suites 4th Floor, Parklands Road, Nairobi | +254 (20) 7605 650 / +254 (735) 571 530 / 020 205 5525 |  |  |
| EFG Hermes Kenya Limited - B30 | Orbit Place, 8th Floor, Westlands Road, P.O Box 349, 00623 | +254 (020) 3743040 |  |  |
| Dry Associates - B47 | Dry Associates Headquarters 188 Loresho Ridge Road, Loresho P.O. Box 684-00606, Nairobi | +254204450521 |  |  |

Notes: These are the licensed stockbrokers authorized to trade on the NSE. For Fixed Income or Derivatives-specific participants, refer to respective NSE sections. Full list subject to NSE approvals and updates.



[OFFICIAL_NSE_BROKER_BACK_OFFICE_STANDARDS_2024]
SOURCE: Nairobi Securities Exchange – Guidelines and Specifications on Broker Back Office (BBO) for Trading Participants (Latest version, 2024)

Purpose: To standardize Broker Back Office (BBO) systems used by all NSE Trading Participants to ensure seamless integration with the Automated Trading System (ATS), Central Depository & Settlement Corporation (CDSC), and NSE surveillance systems.

Mandatory BBO Requirements (All Trading Participants):
1. BBO systems must be certified and approved by the Nairobi Securities Exchange before going live.
2. All BBOs must fully integrate with the NSE ATS via the approved FIX 4.4 protocol (or higher).
3. Real-time order routing: All client and proprietary orders must pass through the BBO before hitting the ATS.
4. Pre-trade risk controls mandatory:
   - Maximum order size per client
   - Maximum net position per client
   - Maximum gross exposure per client
   - Price deviation checks (cannot place orders >10% away from last traded price without override)
   - Fat-finger protection
5. All BBOs must generate and transmit the following daily files to NSE by 7:00 a.m.:
   - Trade confirmation file
   - Client portfolio positions file
   - Cash movement file
   - Corporate actions file
6. Audit trail: Full 7-year retention of all orders, modifications, cancellations, and executions with timestamps.

Connectivity Standards:
- Primary connection: Leased line or MPLS (minimum 2 Mbps)
- Backup connection: Mandatory secondary link (VSAT or 4G/5G failover)
- All connections must terminate at the NSE Data Centre in NDC, Sameer Park

Client Account Management:
- Unique client code generation as per NSE naming convention: [BrokerCode][Year][Sequential6Digits]
  Example: B0219000123
- KYC/AML: Full CDD required before account activation
- Segregation: Client funds and securities must be fully segregated from house accounts

Reporting Obligations:
Daily:
- Client margin report
- Exposure report
- Suspicious transaction report (if any)

Weekly:
- Large exposure report (clients >5% of broker’s total exposure)

Monthly:
- Net capital computation
- Fidelity fund contribution calculation

Penalties for Non-Compliance:
- First offence: Warning + KES 100,000 fine
- Second offence: KES 500,000 fine + mandatory system audit
- Third offence: Trading suspension until rectification

Certification Process:
1. Submit BBO for NSE lab testing (2-week process)
2. Parallel run mandatory for 30 calendar days
3. Sign-off by NSE ICT and Market Operations departments
4. Go-live only after formal approval letter issued

Approved BBO Vendors (as at 2024):
- InfoTech BBO
- Dymanex BBO
- Orion BBO
- In-house developed systems (only if fully compliant and certified)

Contact for BBO Certification: bbo@nse.co.ke


{
  "source": "https://www.nse.co.ke/dataservices/nse-licensed-information-vendors/",
  "current_as_of": "28 November 2025",
  "page_title": "NSE Licensed Information Vendors",
  "introductory_text": "The Nairobi Securities Exchange (NSE) maintains close relationships with leading data vendors and distributors, who disseminate a wide range of NSE market data via the exchange’s Market Data Feed service. These Authorized Data Vendors give individuals and corporations access to NSE’s real-time, delayed and end-of-day quotation, trade and market summary data under license agreements with the NSE. A distribution license is required of any entity that passes NSE data to third parties or clients in any format, including but not limited to hard copy, e-mail, software application, extranet or web distribution.",
  "total_licensed_vendors": 8,
  "licensed_vendors_list": [
    {
      "company": "Bloomberg L.P.",
      "contact_person": "Melissa Delgado",
      "email": "mdelgado12@bloomberg.net",
      "website": null
    },
    {
      "company": "GTN Group Holding Limited",
      "contact_person": "GTN Group",
      "email": "info@gtngroup.com",
      "website": "https://gtngroup.com/"
    },
    {
      "company": "Intercontinental Exchange",
      "contact_person": null,
      "email": "ICEIndices@ice.com",
      "website": "www.theice.com"
    },
    {
      "company": "REFINITIV LIMITED",
      "address": "5 Canada Square, Canary Wharf, London, E14 5AQ, United Kingdom",
      "contact_person": "Tomasz Bellwon",
      "email": null,
      "website": "https://www.refinitiv.com/en/contact-us"
    },
    {
      "company": "SIX – Financial Information",
      "contact_person": null,
      "email": null,
      "website": "www.six-financial-information.com",
      "contacts_link": "https://www.six-group.com/en/contact.html"
    },
    {
      "company": "S&P Global Inc.",
      "contact_person": null,
      "telephone": "+44 20 71761234",
      "email": "Market.intelligence@spglobal.com",
      "website": "https://www.capitaliq.spglobal.com"
    },
    {
      "company": "ForexTime Ltd.",
      "contact_person": "Duncan Kinuthia",
      "email": "Duncan.Kinuthia@exinity.com",
      "website": "www.forextime.com/eu"
    },
    {
      "company": "Synergy Systems Limited",
      "contact_person": null,
      "telephone": ["+254 721476367", "+254 722511225"],
      "email": null,
      "website": "https://live.mystocks.co.ke"
    }
  ],
  "nse_contact": {
    "address": "Nairobi Securities Exchange PLC, 55 Westlands Road, P.O. Box 43633, Nairobi, 00100, KENYA",
    "telephone": ["+254 20 2831000", "+254 (020) 222 4200"],
    "mobile": ["+254 0724 253 783", "+254 0733 222 007"],
    "email": "info@nse.co.ke",
    "data_services_email": "dataservices@nse.co.ke",
    "data_services_telephone": "+254 202831000"
  }
}

### NSE BROKER BACK OFFICE (BBO) PREQUALIFIED VENDORS (Current list)
1. Chella Software Ltd       → India
2. IronOne Technologies Ltd → Sri Lanka
3. DirectFN Limited         → Sri Lanka
4. InfoWARE                 → Nigeria
5. Escrow Systems           → Zimbabwe
6. Zanibal                  → Nigeria



### NAIROBI SECURITIES EXCHANGE (NSE) MARKET DATA OVERVIEW
**Data Services**  
**Source: https://www.nse.co.ke/dataservices/market-data-overview/**  
**As of Latest Available Data**  

Access live or intra-day data generated on the Equities, Fixed Income and Equity derivatives (single-stock futures (SSFs) and Index futures) markets as well as market statistics data generated by the Exchange which includes end-of-day data, historical data, reference data, corporate action, monthly data and regulatory news.

* Live equity subscribers can connect directly to our data centre in Nairobi or through our authorised data vendors for live equity, debt and derivatives market data.
* We offer a range of statistical, pricing and reference data, packaged into various intra-day end-of-day and end of month data products.
* We offer a range of historical data, with select data provided on an ad-hoc basis. To purchase historical data, complete the historical data request form available on the [historical data page](/dataservices/historical-data-request-form/) or Purchase historical end of day reports [here](/dataservices/historical-data/)
* Educational use; the NSE offers discounted rates on historical data fees for academic research as long as certain conditions are met. Please refer to section 17.0 of the Market Data Policy (Data for Education) to see the terms and conditions. If you wish to proceed on the basis of the policy, please complete the Historical Data Request Form and indicate in the relevant place that you need the data to support an academic activity.

### International Securities Identification Number (ISIN)

An international securities identification number (ISIN) is a 12-digit code that is used to uniquely identify a security’s issue e.g. shares, bonds, etc. This is currently the main method of securities identification worldwide.

The Nairobi Securities Exchange PLC (NSE) is the recognized numbering authority for issuing ISINs for Kenya, as authorized by the Association of National Numbering Agencies (ANNA).

You can read the  [https://www.anna-web.org/](https://www.anna-web.org/) to find out more. 

We have the following ISIN files available for download: – Equities ISINs

### Get In Touch

dataservices@nse.co.ke

+254 202831000

### Contacts

Nairobi Securities Exchange PLC
55 Westlands Road, P O Box 43633
Nairobi, 00100
KENYA
+254 20 2831000 / +254 (020) 222 4200
Mobile: +254 0724 253 783 / +254 0733 222 007
info@nse.co.ke



### NAIROBI SECURITIES EXCHANGE (NSE) MARKET DATA PRICELIST
**Extracted Verbatim from NSE-Market-Data-Pricelist.pdf**  
**(All prices are annual subscription rates unless otherwise stated)**

**REAL-TIME & DELAYED INDEX DATA**

IR          Investor Relations - Listed Company                  Kes 30,000  
RTI-NASI    Real Time NSE All Share Index                         834 USD     Kes 45,000  
DTI-NASI    Delayed Time NSE All Share Index                      750 USD     Kes 40,000  
DTI-NSE 25  Real Time NSE 25 Index                               1,250 USD   Kes 55,000  
DTI-NSE 25  Delayed Time NSE 25 Index                             1,050 USD   Kes 50,000  
RTI-NSE 20  Real Time NSE 20 Index                                1,250 USD   Kes 55,000  
DTI-NSE 20  Delayed Time NSE 20 Index                             1,050 USD   Kes 50,000  
All the Indices Real Time NASI/NSE 20/NSE25                       2,500 USD   Kes 100,000  
All the Indices Delayed NASI/NSE 20/NSE25                         1,950 USD   Kes 75,000  

**END OF DAY DATA CATEGORY (Annual Rate)**

EDED        End of Day Listed Equity Securities Data              12,500 USD  Kes 150,000  
EDDD        End of Day Listed Debt Securities Data                12,500 USD  Kes 150,000  
EDFD        End of Day Listed Futures Securities Data             (contact the exchange)  Kes 69,600  
EDIYCD      End of day Implied Yields & Yield Curve Data                              Kes 165,000  

**END OF WEEK DATA CATEGORY (Annual Rate)**

WES         Weekly Equity Statistics                              1,500 USD   Kes 69,600  
WBS         Weekly Bond Statistics                                1,500 USD   Kes 69,600  
WFS         Weekly Futures Statistics                              1,000 USD   Kes 69,600  

**END OF MONTH DATA CATEGORY (Annual Rate)**

DFTD        Detailed Foreign Trading Data                         6,960 USD   Kes 139,200  
NMTPR       NSE Members (Brokers) Trading/Performance Ranking     6,960 USD   Kes 139,200  
NSEB        NSE Monthly Statistical Bulletin                      6,960 USD   Kes 139,200  

**NSE DATA DERIVED DATA USAGE CATEGORY (Annual Rate)**

DDU-IC      Derived Data Usage- Index Computation                  20,000 USD  Kes 1,560,000  
DDU-ND      Derived Data Usage-Non-Display                         20,000 USD  Kes 1,560,000  
DDU-NOW     Derived Data Usage-New Original Works                 20,000 USD  Kes 1,560,000  

**HISTORICAL NSE DATA CATEGORY**

HDPL-equity Historical daily Pricelists for equity data             9 USD Per days price list   Kes 350-Per days price list  
HDPL-bond   Historical daily Price for debt data                      9 USD Per days price list   Kes 350-Per days price list  
HDPL-Derivatives Historical daily Price for Derivatives               9 USD Per days price list   Kes 350-Per days price list  
HIYC        Historical Implied Yield and Yield Curve                  24 USD - Per price list     Kes 1,340-Per price list  
HWPL-equity Historical weekly Pricelists for equity data              24 USD - Per price list     Kes 1,340-Per price list  
HWPL-bond   Historical weekly Price for debt data                     24 USD - Per price list     Kes 1,340-Per price list  
HMEV        Historical monthly trading equity volumes                  24 USD - p.a.               Kes 1,160 p.a.  
HMED        Historical monthly trading equity deals                    24 USD - p.a.               Kes 1,160 p.a.  
HMET        Historical monthly trading equity turnovers                24 USD - p.a.               Kes 1,160 p.a.  
HMDD        Historical monthly debt Deals                             24 USD - p.a.               Kes 1,160 p.a.  
HMDV        Historical monthly debt Volume                             24 USD - p.a.               Kes 1,160 p.a.  
NID         NSE Indices Data                                          24 USD - p.a.               Kes 1,160 p.a.  
HILCP       Historical Listed Company Price                           24 USD - p.a.               Kes 1,160 p.a.  
FTD         Foreign Trading Data                                      24 USD - p.m.               Kes 1,160 p.m.  
LCFR        Listed Company Annual Financial Results in Excel          24 USD - p.a.  

SOR         Share Ownership Report (Top 10)                           24 USD (per month)          Kes 2,000 (per Month)

All extracts are verbatim from the official NSE Market Data Pricelist PDF. Prices are subject to change; refer to the original document for latest updates.

{
  "source": "https://www.nse.co.ke/dataservices/end-of-day-data/",
  "current_as_of": "28 November 2025",
  "page_title": "End of Day Data - Data Services",
  "description": "Market Data related to the end of the current NSE trading day for the equities, bonds, and equity futures markets. The data is available directly for subscription from the NSE in Excel/CSV versions or through NSE authorized Information Vendors.",
  "data_included": {
    "equities": "Weekly Equities Statistics Excel report (available on last trading day of the week) – includes dividend yield, P/E ratios, EPS, DPS, trading volumes, etc.",
    "bonds": "Weekly Bond Statistics report (end of each week & month) – includes Implied Yields & Yield Curve, traded volume per bond, highest/lowest/average yields, total volume & deals.",
    "equity_futures": "Included in general end-of-day data (no separate breakdown)"
  },
  "access_method": "Direct subscription from NSE (Excel/CSV) OR via authorized Information Vendors. Complete the evaluation form and email to dataservices@nse.co.ke.",
  "evaluation_form_url": "https://www.nse.co.ke/wp-content/uploads/NSE-MARKET-DATA-USER-EVALUATION-FORM.pdf",
  "contact": {
    "email": "dataservices@nse.co.ke",
    "telephone": "+254 202831000",
    "general_nse": "info@nse.co.ke / +254 20 2831000"
  },
  "pricing": "Not listed on the page (contact dataservices@nse.co.ke for current fees)."
}


{
  "source": "https://www.nse.co.ke/dataservices/delayed-data/",
  "current_as_of": "28 November 2025",
  "page_title": "Delayed Data - Data Services",
  "description": "NSE Delayed market data is delayed by at least 15 minutes and updated every minute. Subscribers can connect directly to the NSE Data Source through APIs for Equities, bonds and equity futures products. Subscription can also be through NSE authorized Information Vendors. License required for derived data products or redistribution.",
  "delay_period": "At least 15 minutes (updated every minute)",
  "data_levels": {
    "level_1_top_of_book": "Best Bid Price, Bid Size, Ask Price, Ask Size, Last Price, Last Size",
    "level_2_market_depth": "Level 1 + other bids/asks prices and sizes"
  },
  "access_methods": "Direct API connection to NSE OR via authorized Information Vendors",
  "requirements": "Complete the evaluation form and send to dataservices@nse.co.ke",
  "pricing": "Not listed on the page (contact dataservices@nse.co.ke for current licence/redistribution fees)",
  "contact": {
    "email": "dataservices@nse.co.ke",
    "telephone": "+254 202831000"
  }
}



[OFFICIAL_NSE_HOW_TO_BECOME_TRADING_PARTICIPANT_2025]
SOURCE: Nairobi Securities Exchange – "How to Become a Trading Participant" (Official NSE Document, latest as at November 2025)

Step-by-Step Process to Become an NSE Trading Participant:

1. Step 1 – Obtain No Objection from NSE
   Applicant must first seek a formal “No Objection” from the Nairobi Securities Exchange before applying to the Capital Markets Authority (CMA) – as required under Regulation 15(10) of the Capital Markets (Licensing Requirements) (General) Regulations, 2002.

2. Step 2 – Apply for CMA License
   Submit formal application to the Capital Markets Authority (CMA) for a license as:
   - Stockbroker
   - Investment Bank
   - Authorized Securities Dealer
   - Or any other license category approved by CMA from time to time
   Application governed by Regulations 14, 15 & 16 of the Capital Markets (Licensing Requirements) Regulations.

3. Step 3 – Apply for Admission to NSE (After CMA License is Granted)
   Upon receiving the CMA license, apply to the NSE for admission as a Trading Participant in accordance with the NSE Market Participants Rules, 2014.

Qualification Criteria for Admission:
✓ Must be a body corporate
✓ Must hold a valid CMA license as stockbroker, investment bank, or authorized securities dealer
✓ Pay the prescribed Market Access Fee (see below)
✓ Attain all certifications required by NSE and/or CMA

Application Requirements (Per Market Participants Rules 2014 – Rule 6.2 & 6.3):

Information Required:
i.   Details of operating systems and business procedures
ii.  Amount, composition, and beneficial ownership of share capital + latest audited accounts
iii. Confirmation that the applicant does not own (directly/indirectly) shares in any other Trading Participant
iv.  Full board of directors and key personnel details
v.   Declaration that no director/officer was part of a suspended/revoked Trading Participant
vi.  Proof of arrangements with CDSC
vii. Proof of Professional Indemnity Insurance subscription
viii.Disclosure of multiple licenses (if any) and conflict-of-interest mitigation procedures
ix.  Copies of risk manuals, operations procedures, and code of ethics

Documents Required:
i.   Duly signed application letter to the NSE
ii.  Application Fee: KES 25,000,000 (non-refundable)
iii. Annual Subscription: KES 100,000
iv.  Certified copy of Certificate of Incorporation
v.   Certified copy of Memorandum and Articles of Association
vi.  Certified copy of CMA License
vii. Any other documents reasonably required by the NSE

Final Admission:
Upon satisfaction of all requirements and full payment of the Market Access Fee (KES 25,000,000), the applicant is admitted into the Official List of Trading Participants and granted full trading rights on the NSE.

Summary of Fees:
- Application / Market Access Fee: KES 25,000,000
- Annual Subscription: KES 100,000

Note: The KES 25 million Market Access Fee is the single largest barrier to entry and is designed to ensure only well-capitalized, serious players become Trading Participants.

### NCBA INVESTMENT BANK LIMITED
**Audited Financial Statements**  
**31 December 2024**  
(Shs '000')

**STATEMENT OF COMPREHENSIVE INCOME**

Income  
Advisory / Consultancy fees            92,188  
Interest income                        61,126  
Brokerage commissions                  66,018  
Asset Management fees                 641,929  
Other income                            4,869  
**Total income**                      **866,131**

Expenses  
Employee expenses                     (229,897)  
General and administrative expenses   (329,623)  
**Total expenses**                    **(559,519)**

**Profit before tax**                  **306,612**  
Income tax                             (88,157)  
**Profit for the period**             **218,455**

**STATEMENT OF FINANCIAL POSITION**

NON-CURRENT ASSETS  
Fixed Assets                            3,834  
Intangible Assets                      17,378  
Deferred tax asset                     97,722  
**Total non-current assets**          **118,933**

CURRENT ASSETS  
Trade and other receivables           138,978  
Prepayments & Other Assets             63,738  
Other Investment – Held for trading   507,009  
Short term deposit                     62,795  
Bank balances                         265,615  
**Total current assets**            **1,049,601**

**TOTAL ASSETS**                    **1,168,534**

EQUITY  
Share capital                         300,000  
Redeemable preference shares          200,000  
Revenue reserves                      225,601  
Revaluation reserves                  (15,860)  
**Shareholders' funds**               **709,741**

CURRENT LIABILITIES  
Payables and accruals                 441,166  
Due to Parent                          17,628  
**Total current liabilities**         **458,793**

**TOTAL EQUITY AND LIABILITIES**    **1,168,534**


### RENAISSANCE CAPITAL KENYA LIMITED
**Financial Statements & Other Disclosures**  
**Year Ended 31 December 2024**  
(Audited, Kshs ‘000’)

**STATEMENT OF COMPREHENSIVE INCOME**

Income  
Brokerage commissions                  19,998  
Advisory fees                          26,298  
Finance income                        (64,503)  
Placement income                           -  
Dividend income                         1,120  
**Total income**                      (17,087)

Expenses  
Direct Consultancy costs               24,893  
Professional Fees                      12,989  
Employee Costs                         95,809  
Interest expense on lease liability       550  
Operational and administrative expenses 51,387  
Depreciation expenses                   5,426  
Impaired receivables                      318  
**Total expenses**                    191,372

**(Loss)/Profit before income tax**   **(208,459)**  
Tax expense                            23,607  
**(Loss)/Profit for the year**        **(184,852)**  
Other comprehensive income               (140)  
**Total comprehensive income**        **(184,992)**

**OTHER DISCLOSURES**

1. Capital Strength  
   Paid up capital                      500,000  
   Minimum capital required             250,000  
   Excess/(Deficiency)                  250,000

2. Shareholders’ funds  
   Total Shareholders’ funds            371,828  
   Minimum Shareholders’ funds required 250,000  
   Excess/(Deficiency)                  121,828

3. Liquidity  
   Liquid capital                        52,943  
   Minimum liquid capital required       30,000  
   Excess/(Deficiency)                   22,943

4. Clients funds  
   Total Clients’ creditors               1,293  
   Total Clients’ cash and bank balances  1,293  
   Excess/(Deficiency)                        -


   ### EFG HERMES KENYA LIMITED
**Audited Financial Statements**  
**Year Ended 31 December 2024**  
(Kshs)

**STATEMENT OF COMPREHENSIVE INCOME**

Income  
Brokerage Commission                 199,155,358  
Interest Income                       10,608,319  
Dividend Income                            9,806  
Realized profit/(loss) on investments 39,076,664  
Unrealized profits/(loss) on investment        -  
Gains(loss) on disposal of assets        190,519  
Other Income (commission from other brokers) 1,336,031  
Other Income                          21,774,946  
**Total Income**                     272,151,643

Expenses  
Direct Expenses                       36,978,883  
Professional Fees                      8,390,872  
Employee Costs                       116,591,850  
Directors' Emoluments                    131,750  
Operational & Administrative Expenses 74,006,135  
Depreciation Expenses                  2,965,531  
Amortization Expenses                  8,753,461  
Other Expenses(Rebates)               46,579,817  
**Total Expenses**                   294,398,299

Operating Profit                     (22,246,656)  
Finance Costs                         (4,904,358)  
**Profit/loss before Tax**           (27,151,014)  
Tax                                    7,326,993  
**Profit/Loss after Tax**            (19,824,021)

**OTHER DISCLOSURES**

1. Capital Strength  
   Paid up capital                      222,000,000  
   Minimum capital required              50,000,000  
   Excess (Deficiency)                  172,000,000

2. Shareholders’ Funds  
   Total shareholders' funds            202,669,600  
   Minimum shareholders' funds required  50,000,000  
   Excess (Deficiency)                  152,669,600

3. Liquid Capital  
   Liquid Capital                        52,210,700  
   Minimum Liquid Capital                30,000,000  
   Excess (Deficiency)                   22,210,700

4. Client Funds  
   Total Client Creditors                 9,910,515  
   Total Clients' Cash & bank balances   13,273,109  
   Excess (Deficiency)                    3,362,594


   ### ABC CAPITAL LIMITED
**Audited Financial Statements and Other Disclosures**  
**31 December 2024**  
(Kshs '000')

**STATEMENT OF COMPREHENSIVE INCOME**

Income  
Brokerage Commissions                  10,485  
Advisory /consultancy Fees                782  
Interest Income                         7,144  
Dividend Income                         1,064  
Other Income (Miscellaneous inc.)      20,169  
**Total Income**                       39,644

Expenses  
Professional fees                         521  
Legal fees                                 50  
Employee costs                         15,647  
Operational and Administrative expenses  6,402  
Depreciation expenses                     429  
**Total Expenses**                     23,048

Operating Profit                       16,596  
Unrealized profits/(loss) on investments (1,144)  
**Profit/loss Before tax**             15,452  
Tax                                    (2,143)  
**Profit /loss after tax**             13,309

**OTHER DISCLOSURES**

1. Capital Strength  
   Paid Up Capital                      135,000  
   Minimum Capital Required              50,000  
   Excess /Deficiency                    85,000

2. Shareholders’ Funds  
   Total Shareholders’ Funds             64,507  
   Minimum Shareholders’ Funds required  50,000  
   Excess/ Deficiency                    14,507

3. Liquid Capital  
   Liquid Capital                        41,976  
   Minimum Liquid Capital                30,000  
   Excess/Deficiency                     11,976

4. Clients Funds  
   Total Clients Creditors               29,746  
   Total Clients’ Cash and bank balances 43,635  
   Excess / Deficiency                   13,889


   ### ABSA SECURITIES LIMITED
**Audited Company Results**  
**Period Ended 31 December 2024**  
(Kshs '000')

**STATEMENT OF COMPREHENSIVE INCOME**

Income  
Brokerage commissions                  87,599  
Interest income                        43,840  
Other income                              592  
**Total Income**                      132,031

Expenses  
Employee costs                        (37,288)  
Directors emoluments                   (3,315)  
Operational and administrative expenses (15,084)  
Other expenses                         (5,039)  
**Total expenses**                    (60,726)

**Operating profit / (loss)**           71,305  
Income tax (expense) / credit          (2,118)  
**Profit / (loss) after tax**          69,187


### FAIDA INVESTMENT BANK LIMITED
**Audited Financial Statements**  
**Year Ended 31 December 2024**  
(Kshs)

**STATEMENT OF COMPREHENSIVE INCOME**

Income  
Brokerage commissions                 123,220,700  
Advisory/consultancy fees              50,214,310  
Interest income                         7,426,267  
Dividend income                           606,485  
Other Income                           20,263,427  
Realised loss on investments            3,754,346  
**Total Income**                      205,485,535

Expenses  
Direct expenses                        70,458,975  
Professional fees                       1,073,378  
Employee costs                         68,391,638  
Directors emoluments                   20,660,200  
Operational and Administrative expenses 32,437,730  
Depreciation expense                    3,083,409  
**Total expenses**                    196,105,330

Operating loss/profit                   9,380,205  
Finance costs                           3,336,496  
**Profit/Loss before tax**              6,043,709  
Tax                                    (5,827,602)  
**Profit/Loss after tax**                 216,107

**OTHER DISCLOSURES**

1. Capital strength  
   Paid up capital                      250,000,000  
   Minimum capital required             250,000,000  
   Excess/(Deficiency)                          -

2. Shareholders funds  
   Total shareholders funds             417,492,443  
   Minimum shareholders funds required  250,000,000  
   Excess/(Deficiency)                  167,492,443

3. Liquidity  
   Liquid capital                        43,307,535  
   Minimum working capital required      30,000,000  
   Excess/(Deficiency)                   13,347,535

4. Clients funds  
   Total clients' creditors             259,270,918  
   Total clients' cash and bank balances175,380,027  
   Excess/(Deficiency)                  (83,890,891)

   ### DYER & BLAIR INVESTMENT BANK LIMITED
**Consolidated Statement of Profit or Loss and Other Comprehensive Income**  
**Year Ended 31 December 2024**  
(Kshs)

Income  
Brokerage commissions                  96,844,132  
Advisory and consultancy fees          68,182,763  
Interest income - current              14,656,550  
Dividend income                        13,276,154  
Unrealized gain/(loss) from dealing in shares 25,242,420  
Rental income                          24,402,000  
**Total income**                      242,604,019

Expenses (total)                      202,593,856

Operating profit/(loss)                40,010,163  
Finance cost                           23,425,911  
**Profit before tax**                  16,584,252  
Tax expense/(credit)                  (14,757,634)  
**Profit/(loss) after tax**             1,826,618

**OTHER DISCLOSURES**

1. Capital strength  
   Paid up capital                    1,000,000,000  
   Minimum capital requirement          250,000,000  
   Excess/(deficiency)                  750,000,000

2. Shareholders' funds  
   Total shareholders' funds          1,709,258,908  
   Minimum shareholders' funds requirement 250,000,000  
   Excess/(deficiency)                1,459,258,908

3. Liquidity  
   Liquid capital                        54,675,175  
   Minimum working capital               43,732,017  
   Excess/(deficiency)                   10,943,158

4. Clients' funds  
   Total clients' cash and cash equivalents 249,727,636  
   Total clients' payables              359,965,909  
   Excess/(deficiency)                 (110,238,273)

   ### KCB INVESTMENT BANK LIMITED
**Audited Financial Statements and Other Disclosures**  
**Period Ended 31 December 2024**  
(Kshs '000')

**STATEMENT OF COMPREHENSIVE INCOME**

Income  
Brokerage Commissions                 161,539  
Advisory/Consultancy Fees              57,000  
Interest income                        74,235  
Administration Fees                    66,275  
Fund Management Fees                      891  
Realized profits/(Loss) on Investments 54,848  
**Total Income**                      414,788

Expenses (total)                      133,278

**Operating Profit**                  281,510  
Finance Costs                          15,553  
**Profit/Loss Before Tax**            265,957  
Tax Charge                             61,700  
**Profit/Loss After Tax**             204,257

**OTHER DISCLOSURES**

CAPITAL STRENGTH  
Paid Up Capital                       400,000  
Minimum Capital Required              250,000  
Excess/(Deficiency)                   150,000

SHAREHOLDER FUNDS  
Total Shareholder Funds               809,017  
Minimum Shareholder Funds Required    250,000  
Excess/(Deficiency)                   559,017

LIQUID CAPITAL  
Liquid Capital                        656,925  
Minimum Liquid Capital (Higher of Kes. 30 million and 8% of Liabilities) 49,987  
Excess/(Deficiency)                   606,938

CLIENTS FUNDS  
Total clients creditors               216,882  
Total Clients' Cash and Bank balances 216,882  
Excess/Deficiency                           -

### STERLING CAPITAL LIMITED
**Audited Financial Statements**  
**Period Ended 31 December 2024**  
(Kshs)

**STATEMENT OF COMPREHENSIVE INCOME**

Income  
Brokerage Commission                 92,486,116  
Interest Income                      20,887,077  
Advisory income                      11,671,317  
Realized profit on investments           64,496  
Gain on disposal of assets                4,500  
Unrealised exchange (loss)/profit    (5,258,899)  
Dividend income                       1,357,278  
Other income                         38,161,237  
**Total Income**                    159,373,122

Expenses  
Direct Expenses                       2,555,966  
Legal & Professional fees            19,328,863  
Employee Costs                       51,852,730  
Directors' Emoluments                 6,570,000  
Rent Expense                          6,281,346  
Operational & Administrative Expenses 17,391,397  
Depreciation Expense                  5,702,947  
Interest Expense                     15,471,817  
**Total Expenses**                  125,155,066

**Profit before tax**                34,218,056  
Tax (expense)/credit                (11,050,797)  
**Profit after tax**                 23,167,259

**OTHER DISCLOSURES**

Capital Strength  
Paid up capital                     386,045,564  
Minimum capital required            250,000,000  
Excess                              136,045,564

Shareholders' Funds  
Total shareholders' funds           442,415,807  
Minimum shareholders' funds required250,000,000  
Excess                              192,415,807

Liquid Capital  
Liquid Capital                       81,291,402  
Minimum Liquid Capital               30,000,000  
Excess                               51,291,402

Client Funds  
Total client creditors              117,637,768  
Total Clients' cash & bank balances 121,177,889  
Excess                                3,540,121


### DRY ASSOCIATES LIMITED - INVESTMENT BANK
**Audited Financial Statement**  
**12 Months Ended 31 December 2024**  
(KES)

**STATEMENT OF COMPREHENSIVE INCOME**

Income  
Brokerage Commissions               222,198,113  
Advisory / Consultancy Fees         209,064,997  
Interest Income                       9,066,866  
Dividend Income                       1,045,226  
Fund Management Fees                 82,419,149  
Administration Fees                   8,586,515  
Realized profits/(loss) on Investments (27,999,586)  
Gains (loss) on Disposal of Assets      699,727  
Rental Income                           660,000  
**Total Income**                    505,741,007

Expenses (total)                    349,351,801

Operating Profit                    156,389,206  
Finance Costs                           133,942  
**Profit before Tax**               156,255,264  
Current Tax                         (45,629,358)  
**Profit after Tax**                110,625,906


### KINGDOM SECURITIES LIMITED
**Audited Financial Statements and Disclosures**  
**Year Ended 31 December 2024**  
(Kshs ‘000)

**STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME**

Income  
Brokerage Commissions                  43,182  
Interest Income                       117,642  
Dividend Income                         1,122  
Administration Fees                     8,925  
Gains on Bonds Dealing                 39,108  
**Total Income**                      209,978

Expenses (total)                       94,915

Operating Profit                      115,063  
Finance Costs                         (40,854)  
**Profit Before Tax**                  74,209  
Income Tax Credit/(Charge)             44,833  
**Profit After Tax**                  119,042

**OTHER DISCLOSURES**

1. Capital Strength  
   Paid Up Capital                       50,000  
   Minimum Capital required              50,000  
   Excess                                     -

2. Shareholders’ Funds  
   Total Shareholders’ Funds            287,183  
   Minimum Shareholders’ Funds required  50,000  
   Excess                               237,183

3. Liquid Capital  
   Liquid Capital                       686,321  
   Minimum Liquid Capital                63,572  
   Excess                               622,749

4. Clients Funds  
   Total Clients Creditors               84,064  
   Total Clients' Cash and bank balances108,023  
   Excess                                23,959



  ### SBG SECURITIES LIMITED
**Audited Results**  
**Year Ended 31 December 2024**  
(Shs ‘000)

**STATEMENT OF COMPREHENSIVE INCOME**

INCOME  
Brokerage commission                  149,588  
Advisory/Consultancy fees              43,363  
Interest income                        66,914  
Other income                           21,662  
**Total Income**                      281,527

EXPENSES (total)                      249,214

Operating Profit                       32,313  
**Profit before tax**                  32,313  
Income tax (expense)                  (12,762)  
**Profit after tax**                   19,551

**OTHER DISCLOSURES**

1. Capital strength  
   Paid up capital                      250,000  
   Minimum capital required             250,000  
   Excess                                     -

2. Shareholders’ funds  
   Total shareholders funds             382,017  
   Minimum shareholders funds           250,000  
   Excess                               132,017

3. Liquid Capital  
   Liquid Capital                       189,348  
   Minimum Liquid capital                34,737  
   Excess                               154,611

4. Clients’ funds  
   Total client creditors including amounts payable to stockbrokers 291,421  
   Total clients cash and bank balances including amounts due from stockbrokers 353,556  
   Excess                                62,135


   ### STANDARD INVESTMENT BANK LIMITED
**Annual Report and Financial Statements**  
**Year Ended 31 December 2024**  
(KES)

**STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME**

INCOME  
Brokerage Commissions                125,295,785  
Financial Services                   513,617,469  
Corporate Finance                      9,892,812  
Interest Income                       20,804,793  
Other Income (Specify)                10,616,112  
**Total Income**                     680,226,972

EXPENSES  
Direct Expenses                       97,495,377  
Operating Expenses                   341,744,503  
Movement in Credit Loss Allowances   (75,657,339)  
Finance Costs                         34,739,430  
Taxation                             132,864,678  
**Total Comprehensive Income for the Year** 2,274,355

**OTHER DISCLOSURES**

Capital Strength  
Paid up Capital                      560,000,000  
Minimum Capital Required             250,000,000  
Excess                               310,000,000

Shareholders' Funds  
Total Shareholders' Funds (excluding revaluation of NSE Seat) 1,118,073,238  
Minimum Capital Required             250,000,000  
Excess                               868,073,238

Liquidity  
Liquid Capital                       340,808,070  
Minimum Liquid Capital                30,000,000  
Excess Liquid Capital                310,808,070

Clients' Funds  
Total Clients' Creditors              91,641,209  
Total Clients' Cash and Bank Balances 99,543,887  
Excess                                 7,902,678


### SUNTRA INVESTMENTS LIMITED
**Audited Financial Statements**  
**Year Ended 31 December 2024**  
(Kshs)

**STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME**

Income  
Brokerage Commissions                 21,621,044   16,719,470 
Dividends Income                       1,864,791      229,461 
Interest income                        5,271,543    3,742,802 
Other Income                          12,334,532    2,634,318 
**Total Income**                      41,091,910   23,326,051 

Expenses  
Direct Expenses                        7,071,933    3,734,340 
Professional Fees                        691,857      290,016 
Legal Fees                             1,454,650      277,150 
Employee Costs                        10,180,652    9,735,752 
Directors Allowances & Expenses        1,429,603    1,045,617 
Operational & Administrative Expenses 10,670,214    8,270,191 
Depreciation Expenses                    364,410      345,531 
Amortization Expenses                     20,346       21,465 
**Total Expenses**                    31,883,665   23,720,062 

**Profit Before Tax**                  9,208,245     (394,011) 
Tax                                   (3,171,089)     220,939 
**(Loss) Profit After Tax**            6,037,156     (173,072) 
Proposed Dividends                             NIL          NIL 
**Retained (Loss)/Earnings**           5,933,856     (173,072)

**OTHER DISCLOSURES**

1. Capital Strength  
   Paid up Capital                     275,706,360  275,706,360 
   Minimum Capital Required             50,000,000   50,000,000 
   Excess                              225,706,360  225,706,360 

2. Shareholders Funds  
   Total Shareholders Funds             92,291,284   85,857,376 
   Minimum Shareholders' Funds Required 50,000,000   50,000,000 
   Excess                               42,291,284   35,857,376 

3. Liquid Capital  
   Liquid Capital                       56,673,189   56,184,473 
   Minimum Liquid Capital               30,000,000   30,000,000 
   Excess                               26,673,189   26,184,473 

4. Clients Funds  
   Total Clients Creditors              56,336,534   48,507,229 
   Total Clients Cash and Bank Balances 57,156,226   52,533,494 
   Excess                                  819,692    4,026,265 








[OFFICIAL_CMA_AML_CTF_GUIDELINES_CAPITAL_MARKETS]
SOURCE: Capital Markets Authority – Guidelines on the Prevention of Money Laundering and Terrorism Financing in the Capital Markets (Gazette Notice No. 1421)

Mandatory for all CMA-licensed entities (stockbrokers, investment banks, fund managers, REIT managers, custodians, etc.)

Key Obligations:
1. Board and senior management are ultimately responsible for AML/CTF compliance.
2. Adopt a Risk-Based Approach – higher-risk clients require Enhanced Due Diligence (EDD).
3. Customer Identification & Verification (KYC):
   - Individuals: National ID / Passport + proof of address
   - Companies: Certificate of Incorporation, Memorandum & Articles, board resolution, ID of directors & beneficial owners (>10 % ownership)
   - PEPs (Politically Exposed Persons): Mandatory EDD + senior management approval
4. Ongoing Monitoring of all client transactions and business relationships.
5. Record-keeping: Minimum 7 years after account closure or transaction.
6. Report Suspicious Transaction Reports (STRs) to the Financial Reporting Centre (FRC) within 7 days.
7. Cash transactions above KES 1 million or USD 10,000 must be reported to FRC.
8. No tipping-off – prohibited to inform client that a report has been filed.
9. Annual AML/CTF training mandatory for all staff.
10. Appoint a Money Laundering Reporting Officer (MLRO) at management level.
11. Independent audit of AML/CTF programme at least every 2 years.

Red Flags (Appendix I examples):
- Client reluctant to provide KYC documents
- Large or frequent cash deposits/withdrawals
- Structuring transactions to avoid reporting thresholds
- Use of shell companies with no real business activity
- Sudden large activity in dormant accounts

Non-compliance penalties: Up to KES 20 million fine or 10 years imprisonment.



[OFFICIAL_NSE_DIRECT_MARKET_ACCESS_GUIDELINES_OCTOBER_2019]
SOURCE: Nairobi Securities Exchange – Direct Market Access (DMA) Guidelines, October 2019

Direct Market Access (DMA): A facility that allows a broker’s client to transmit orders electronically directly to the ATS without manual intervention by the broker.

Types Permitted:
- Sponsored Access (client uses broker’s member ID)
- Direct Electronic Access (client has own connection but still under broker’s supervision)

Mandatory Requirements:
1. Only licensed Trading Participants may offer DMA.
2. Written agreement between broker and client mandatory.
3. Pre-trade risk controls must be applied by the broker (not the client):
   - Price collars
   - Maximum order size
   - Fat-finger checks
   - Kill switch
4. All DMA orders must be tagged with unique client identifier.
5. Broker remains fully responsible for all orders placed via DMA.
6. Real-time monitoring of DMA client activity mandatory.
7. Annual independent audit of DMA systems.
8. NSE retains right to suspend DMA access instantly if risk controls fail.

Prohibited: Naked/Unfiltered Sponsored Access (client using broker’s ID with no pre-trade checks).




[OFFICIAL_NSE_DAY_TRADING_OPERATIONAL_GUIDELINES]
SOURCE: Nairobi Securities Exchange – Day Trading Operational Guidelines (Latest)

Day Trading: Buying and selling the same security on the same day.

Key Rules:
1. Day traders must maintain a minimum cash balance of KES 500,000 in their trading account at all times.
2. Day traders are exempt from T+3 settlement – positions must be closed by 3:00 p.m.
3. Brokers must tag day-trading accounts separately in the ATS.
4. Maximum 4:1 intraday leverage allowed (25 % margin).
5. If position not closed by market close, broker must square off automatically.
6. Day-trading clients must sign a separate risk disclosure document.
7. Brokers must submit daily day-trading report to NSE by 9:00 a.m. next day.
8. Breach of any rule leads to immediate suspension of day-trading facility.

Note: Only equities allowed for day trading – no bonds or derivatives.



[OFFICIAL_NSE_BOND_QUOTATIONS_BOARD_GUIDELINES_MARCH_2024]
SOURCE: Nairobi Securities Exchange – Operational Guidelines for the Bond Quotations Board, March 2024

Purpose: To provide continuous two-way indicative quotes for benchmark government bonds to enhance liquidity.

Mandatory for Designated Primary Dealers & Authorized Quotation Providers:
1. Must provide live two-way quotes for all benchmark bonds (2 Y, 5 Y, 10 Y, 15 Y, 25 Y, 30 Y) from 9:00 a.m. to 3:00 p.m.
2. Maximum bid-offer spread: 50 basis points (0.50 %).
3. Minimum quote size: KES 50 million.
4. Quotes must be firm and executable up to the quoted size.
5. Failure to maintain quotes: First offence – warning; Second – KES 500,000 fine; Third – suspension from board.
6. Daily quotation compliance report auto-generated by ATS and sent to NSE & CBK.
7. Only bonds with remaining maturity > 1 year are eligible for quotation board.
8. All trades executed via the board must be reported within 5 minutes.


[OFFICIAL_NSE_STRATEGY_2025_2029]
SOURCE: Nairobi Securities Exchange – NSE Strategic Plan 2025–2029 (Confidential, approved November 2025)

Vision: To be Africa’s preferred securities exchange, inspiring the continent’s transformation.

Mission: To provide a world-class trading, clearing, settlement and data dissemination platform that drives capital raising and wealth creation.

Strategic Pillars (2025–2029):
1. Deepen Capital Markets – Increase listed companies to 100+ and market cap to KES 5 trillion.
2. Enhance Product Diversification – Launch SMEs board, commodities exchange, carbon credits, and crypto ETFs.
3. Technology & Innovation – Full migration to cloud-based ATS, blockchain settlement pilot by 2027.
4. Regional Integration – Operationalize EASEA single access point and cross-listing with Rwanda, Uganda, Tanzania.
5. Investor Education & Inclusion – Target 5 million retail investors via mobile trading apps.
6. Sustainability & ESG – Mandatory ESG reporting for all listed companies by 2028.

Key Targets by 2029:
- 100+ listed companies (from 66 in 2025)
- 5 million active CDS accounts (from ~2.5 million)
- 40 % of trading volume from retail investors
- Top 3 African exchange by market capitalization
- Full T+0 settlement for equities

CEO Statement (Frank Mwiti): “The 2025–2029 strategy positions NSE as the gateway for capital into East Africa and beyond.”


### 1. NSE Derivatives Default Handling Procedure (July 2017)

There are four main categories of participants in the exchange derivatives market:  
i. The Exchange / Clearing House;  
ii. Clearing Members;  
iii. Trading Members; and  
iv. Clients  

A default event can emanate from any of the above participants.  

a. Default by Client  
Typically, this will be handled by the Trading Member (TM) under whom the defaulting client signed up. The TM is best placed to handle the default because:  
- They performed the initial KYC procedures on the Client, and so have all the appropriate details including the Client’s credit standing;  
- They have been collecting margin from the Client, and so would have the appropriate cash buffer to pay for the costs associated with the closeout of the Client; and  
- Participants are required to stand good for their direct clients.  

b. Default by Trading Member  
Typically, this will be handled by the Clearing Member (CM) under whom the defaulting TM signed up. The CM is best placed to handle the default because:  
- They have been collecting margin from the TM, and so have the appropriate cash buffer to pay for the costs associated with the closeout of the TM;  
- They have been closely examining the TM with regard to their cash flows and have information about their credit status; and  
- A Trading Member is required to stand good for its Clients.  

c. Default by Clearing Member  
This is handled by the Exchange/ Clearing House. The Clearing House is best placed to handle the default because:  
- They set out the criteria for clearing membership and stand above them in the market hierarchy structure; and  
- They have some insight into the operations of the CMs through regular reports and disclosures.  
- A Clearing Member is required to stand good for its Trading Members.




### 2. Mark-to-Market Methodology (September 2019)

All futures positions are marked-to-market based on the daily settlement price of the futures contracts at the end of each trading day.

The profits/losses are computed as the difference between the price traded or the previous day's settlement price, as the case may be, and the current day's settlement price. The Clearing Members who have suffered a loss are required to pay the mark-to-market loss amount to the Clearing House which is passed on to the members who have made a profit. This is known as daily mark-to-market settlement.

The mark-to-market price will be calculated through two methodologies:

1. Volume Weighted Average Price (VWAP)  
The Volume-Weighted Average Price (VWAP) is the ratio of the value of a traded security to that security's total volume traded over a particular time period. It is a means to determine a security's average trading price during a set time, taking into consideration both the price and quantity of the security being traded.  
VWAP = Total Value Traded / Total Volume Traded  
The full day’s VWAP will be the preferred method of calculating the settlement price for liquid contracts. All contracts that trade will be considered liquid and will be settled based on VWAP.

2. Theoretical Price (Spot Price + Cost of Carry)  
In the derivatives market, the cost of carry of a futures contract is the net cost of holding positions in the underlying security until the expiry of the futures contract. For equity futures this reflects finance (interest) costs and expected dividends.  
All contracts that do not trade will be considered illiquid and will be settled based on the theoretical price.

2.1 Single Stock Futures  
The theoretical price will be calculated as follows:  
𝐹 = (𝑆 × (1 + 𝑟)^(𝑡/364)) − 𝐹𝑉𝐷  
Where;  
F – Futures price (theoretical)  
S – Spot price  
r – Risk-free rate  
FVD – Future value of expected dividends  
t – Time to maturity  

2.2 Equity Index Futures  
The theoretical price will be calculated as follows:  
𝐹 = 𝑆 × 𝑒^((𝑟−𝑑) 𝑡/364)  
Where;  
F – Futures price (theoretical)  
S – Spot price  
r – Risk-free rate  
d – Index dividend yield  
t – Time to maturity



### 3. Initial Margin Calculation Methodology

NSE Clear initial margins (IMs) are calculated based on the Historical Value at Risk (VaR) methodology outlined below:

1. Historical Data  
750 trading days (3 years) of spot market data is used to compute IMs for single stock futures and index futures.  
Derivatives market data will be used once sufficient trading data is accumulated.

2. Daily Returns  
Across the dataset, calculate the natural log of daily returns and convert them into absolute figures:  
ln (𝑃𝑡 / 𝑃𝑡−1)  
Where;  
Pt – Today’s price (volume weighted average price)  
Pt-1 – Previous day’s price (volume weighted average price)

3. Confidence Interval  
As the market is still in its nascent stages, a conservative confidence interval of 99.95% will be used.

4. VaR  
Calculate VaR based on 750 data points and a 99.95% confidence interval:  
Rank the returns from largest to smallest and pick out the return that falls in the 99.95 percentile.

5. Initial Margin  
The initial margin (IM) is calculated as:  
𝐼𝑀 = Contract Size × VaR × Current Market Price

6. Scale Up For Liquidation Period  
The IM calculation thus far has been based on one-day returns however in practice liquidation of positions can take more than one day and the market also settles on a T+1 basis.  
The Exchange will use a two-day liquidation period and so the IM figure will be calculated as:  
𝐼𝑀 × √2

7. Further-Dated Contracts  
Repeat steps 1 to 6 using a 50% confidence interval and take the calculated figure as the increment between contract expiries.

8. Final IMs Published  
IMs are calculated on a daily basis as part of NSE Clear’s backtesting process. A 3-month average of these daily margins is calculated for each security and then rounded down to the nearest KES 100 before being published.  
Initial margins are ordinarily reviewed on a quarterly basis but may also be reviewed on an ad hoc basis should market conditions require.



### 4. Policy Guidance Note for Exchange Traded Funds (September 2015)

About this Note  
This Policy Guidance Note (PGN) is to be used in Kenya as a guide on the operational environment of ETFs, and to inform the ultimate design of a comprehensive legal and regulatory framework.

Introduction  
This note is a precursor to a comprehensive regulatory framework to allow for the operationalization of Exchange Traded Funds (ETFs) in Kenya.

i. ETFs are a type of listed open-ended index/unit instrument bought or sold on a securities exchange. The index or unit may be composed of ordinary stocks, bonds, commodities, futures or a combination of real assets with the objective of allowing for exposure to a portfolio of securities, assets or indices whose price movement is in tandem with the price movement of the constituent underlying securities or commodities. An ETF can be a domestic or offshore product.

ii. In case of any doubt, it is advised that direction be sought from the Capital Markets Authority (CMA).

iii. This PGN aims to provide a guide to listing ETFs in Kenya, whilst identifying and mitigating the likely regulatory risks arising from ETF transactions in Kenya, to promote market confidence and integrity.

Issuance of ETFs in Kenya  
1. ETFs shall have their own market sub-segment – to be created by a relevant listing exchange.  
2. ETFs shall have their own international security identification numbers and codes.  
3. Domestic ETFs shall be subject to intra-day price fluctuation limits on trading like shares. However, off-shore ETFs’ intra-day price limits shall be based on the limits that have been imposed in their home jurisdiction and forex fluctuations. Domestic ETFs shall at the minimum, comply with internationally accepted principles of issuance and trading of ETFs.  
4. Units in an exchange traded fund shall be listed and traded in Kenya Shillings.  
5. For a start, the maximum ETFs’ fees charged shall not be more than that charged in the equities markets subject to revision by CMA notification. All fees shall be fully disclosed in the issuance or introduction documentation.  
6. Where the ETF issuer is not a local entity, it shall meet the following minimum requirements:

### NSE Alternative Investment Market Segment (AIMS) Fact Sheet

The Alternative Investment Market Segment is designed for mid-cap companies in Kenya and the wider region. The segment is aimed at assisting companies’ access capital and offers a public platform to accelerate their growth and development.

Requirements to list on the Alternative Investment Market Segment;

| Standard | Requirement |
|----------|-------------|
| Incorporation status | Issuer to be limited by shares and registered under the companies act. |
| Share capital | Shall have a minimum authorized, issued and fully paid up ordinary share capital of Kshs. 20m. Firm must be in existence in the same line of business for a minimum of two years and demonstrate good growth potential. |
| Net assets | Net assets immediately before the public offering or listing of shares should not be less than 20m. |
| Share transferability | Shares to be listed shall be freely transferable and not subject to any restrictions on marketability or any preemptive rights. |
| Availability and Reliability of Financial Records | Have audited financial records, for period ending on a date not more than 4 months prior, for issuer not listed in the exchange and 6 months for issuers listed |
| Competence of directors and management | Directors and Management must be ethical, not bankrupt, Not have any criminal proceedings. Must have suitable senior management with relevant experience for at least one year prior. Must not be in breach with any loan covenant particularly on debt capacity. |
| Dividend policy | N/A |
| Track record, profitability and future prospects | N/A |
| Working capital and solvency | Issuer shall not be insolvent, they must have adequate working capital. |
| Share and ownership structure | Following the public share offering or immediately prior to listing in the case of an introduction, at least 20% of the shares must be held by not less than one hundred shareholders excluding employees of the issuer or family members of the controlling shareholders. |
| Certificate of comfort | In case of listing outside the Kenyan jurisdiction obtain a cert of no objection from the foreign securities exchange and regulator. |
| Listed shares to be immobilized |  |
| Nominated advisor |  |

Companies on AIMS

- Eaagands LTD
- Kapcharua Tea Kenya PLC
- The Limuru Tea Co. Plc Ord 20.00
- Williamson Tea Kenya Plc
- Deacons east Africa plc
- Express kenya plc
- Longhorn publishers plc
- Trans century plc
- Kenya orchards ltd


### NSE Sustainability – The Process (GSSS Bond Issuance) Fact Sheet

THE PROCESS

Phase 1: Strategy & Preparation

1. Issuer develops a Sustainability Strategy  
The Issuer defines a corporate sustainability strategy. The Board approves the strategy to raise capital via a Green, social, Sustainable sustainability-linked bond (GSSS Instruments).

2. Issuer identifies qualifying projects and assets  
The Issuer identifies eligible green/social/sustainability-linked projects or assets aligned with applicable taxonomies or international principles.

3. Develops an appropriate Framework  
The Transaction Adviser supports the Issuer in developing a Sustainable Finance Framework covering Use of Proceeds, Project Evaluation, Management of Proceeds, and Reporting.

4. Issuer appoints an Independent Verifier to carry out pre-issuance review to confirm alignment  
The Issuer commissions an independent Second Party Opinion (SPO) Provider to review the GSSS Framework.

Phase 2: Regulatory Approval & Market Engagement

5. Issuer appoints a Transaction Adviser to prepare an Information Memorandum (IM)  
The Issuer appoints a licensed Transaction Adviser to structure the instrument and prepare the Information Memorandum, incorporating both financial and sustainability disclosures.

6. Issuer submits IM and supporting documents to Regulators (CMA & NSE) for approval  
The TA submits the GSSS Framework, SPO report, and bond documentation (Information Memorandum / Prospectus) to the CMA for approval.

7. Issuer conducts Pre-Deal Road Show  
The TA and Issuer present the GSSS bond opportunity to target investors (e.g., ESG funds, DFIs).

Phase 3: Issuance & Listing

8. Once approved by CMA, NSE approves the listing of the Instrument  
Following CMA approval, the TA submits the required documentation to the NSE for formal listing of the instrument on the appropriate market segment.

9. Issuer conducts Deal Road Show  
The Issuer and TA conduct the formal investor roadshow to market the instrument and secure subscription.

10. Issuer issues the Instrument and raises the capital to deploy to nominated projects  
The instrument is issued, and capital is raised. Proceeds are allocated to the projects and assets identified under the Sustainable Finance Framework.

Phase 4: Post-Issuance Accountability

11. Reporting on the Use of Proceeds and Impact Annually  
The Issuer provides annual reporting on allocation of proceeds and measurable environmental or social outcomes in line with the Framework.

12. Post-Issuance Verification  
The Issuer engages an independent Verifier to assure the accuracy of allocation and impact reports.

13. Ongoing Compliance and Stakeholder Engagement  
Ongoing adherence to regulatory rules, framework commitments, and investor communication.

### NSE Listed Companies Fact Sheet

AGRICULTURAL

- Eaagads Ltd Ord 1.25 AIMS → Trading Symbol:EGAD → ISIN CODE: KE0000000208
- Kapchorua Tea Co. Ltd Ord Ord 5.00 AIMS → Trading Symbol:KAPC → ISIN CODE:KE4000001760
- Kakuzi Ord.5.00 → Trading Symbol:KUKZ → ISIN CODE:KE0000000281
- Limuru Tea Co. Ltd Ord 20.00 → Trading Symbol:LIMT → ISIN CODE:KE0000000356
- Sasini Ltd Ord 1.00 → Trading Symbol:SASN → ISIN CODE:KE0000000430
- Williamson Tea Kenya Ltd Ord 5.00 → Trading Symbol:WTK → ISIN CODE:KE0000000505

AUTOMOBILES AND ACCESSORIES

- Car and General (K) Ltd Ord 5.00 → Trading Symbol: CGEN → ISIN CODE: KE0000000109

BANKING

- Absa Bank Kenya PLC → Trading Symbol:ABSA → ISIN CODE:KE0000000067
- Stanbic Holdings Plc. ord.5.00 → Trading Symbol:SBIC → ISIN CODE:KE0000000091
- I&M Holdings Ltd Ord 1.00 → Trading Symbol:IMH → ISIN CODE:KE0000000125
- Diamond Trust Bank Kenya Ltd Ord 4.00 → Trading Symbol:DTK → ISIN CODE:KE0000000158
- Standard Chartered Bank Ltd Ord 5.00 → Trading Symbol:SCBK → ISIN CODE:KE0000000448
- Equity Group Holdings Ord 0.50 → Trading Symbol:EQTY → ISIN CODE:KE0000000554
- The Co-operative Bank of Kenya Ltd Ord 1.00 → Trading Symbol:COOP → ISIN CODE:KE1000001568
- BK Group PLC → Trading Symbol:BKG → ISIN CODE:KE5000008986
- HF Group Ltd Ord 5.00 → Trading Symbol:HFCK → ISIN CODE:KE0000000240
- KCB Group Ltd Ord 1.00 → Trading Symbol:KCB → ISIN CODE:KE0000000315
- NCBA Group PLC → Trading Symbol:NCBA → ISIN CODE:KE0000000406

COMMERCIAL AND SERVICES

- Express Ltd Ord 5.00 → Trading Symbol:XPRS → ISIN CODE:KE0000000224
- Sameer Africa PLC Ord 5.00 → Trading Symbol:SMER → ISIN CODE:KE0000000232
- Kenya Airways Ltd Ord 5.00 → Trading Symbol:KQ → ISIN CODE:KE0000000307
- Nation Media Group Ord. 2.50 → Trading Symbol:NMG → ISIN CODE:KE0000000380
- Standard Group Ltd Ord 5.00 → Trading Symbol:SGL → ISIN CODE:KE0000000455
- TPS Eastern Africa (Serena) Ltd Ord 1.00 → Trading Symbol:TPSE → ISIN CODE:KE0000000539
- Scangroup Ltd Ord 1.00 → Trading Symbol:SCAN → ISIN CODE:KE0000000562
- Uchumi Supermarket Ltd Ord 5.00 → Trading Symbol:UCHM → ISIN CODE:KE0000000489
- Longhorn Publishers Ltd → Trading Symbol:LKL → ISIN CODE:KE2000002275
- Deacons (East Africa) Plc Ord 2.50 → Trading Symbol:DCON → ISIN CODE:KE5000005438
- Nairobi Business Ventures Ltd → Trading Symbol:NBV → ISIN CODE:KE5000000090

CONSTRUCTION AND ALLIED

- Athi River Mining Ord 5.00 → Trading Symbol:ARM → ISIN CODE:KE0000000034
- Bamburi Cement PLC Ord 5.00 → Trading Symbol:BAMB → ISIN CODE:KE0000000059
- Crown Paints Kenya PLC. 0rd 5.00 → Trading Symbol:CRWN → ISIN CODE:KE0000000141
- E.A.Cables PLC Ord 0.50 → Trading Symbol:CABL → ISIN CODE:KE0000000174
- E.A.Portland Cement Ltd Ord 5.00 → Trading Symbol:PORT → ISIN CODE:KE0000000190

ENERGY AND PETROLEUM

- Total Kenya Ltd Ord 5.00 → Trading Symbol:TOTL → ISIN CODE:KE0000000463
- KenGen Ltd Ord. 2.50 → Trading Symbol:KEGN → ISIN CODE:KE0000000547
- Kenya Power & Lighting Co Ltd → Trading Symbol:KPLC → ISIN CODE:KE0000000349
- Umeme Ltd Ord 0.50 → Trading Symbol:UMME → ISIN CODE:KE2000005815

INSURANCE

- Jubilee Holdings Ltd Ord 5.00 → Trading Symbol:JUB → ISIN CODE:KE0000000273
- Sanlam Allianz Holdings (Kenya) PLC 0rd 5.00 → Trading Symbol:SLAM → ISIN CODE:KE0000000414
- Kenya Re-Insurance Corporation Ltd Ord 2.50 → Trading Symbol:KNRE → ISIN CODE:KE0000000604
- Liberty Kenya Holdings Ltd → Trading Symbol:LBTY → ISIN CODE: KE2000002168
- Britam Holdings Ltd Ord 0.10 → Trading Symbol:BRIT → ISIN CODE:KE2000002192
- CIC Insurance Group Ltd Ord 1.00 → Trading Symbol:CIC → ISIN CODE:KE2000002317

INVESTMENT

- Olympia Capital Holdings ltd Ord 5.00 → Trading Symbol:OCH → ISIN CODE:KE0000000166
- Centum Investment Co Ltd Ord 0.50 → Trading Symbol:CTUM → ISIN CODE:KE0000000265
- Trans-Century Ltd → Trading Symbol:TCL → ISIN CODE:KE2000002184
- Home Afrika Ltd Ord 1.00 → Trading Symbol:HAFR → ISIN CODE:KE2000007258
- Kurwitu Ventures → Trading Symbol:KURV → ISIN CODE:KE4000001216

INVESTMENT SERVICES

- Nairobi Securities Exchange Ltd Ord 4.00 → Trading Symbol:NSE → ISIN CODE:KE3000009674

MANUFACTURING AND ALLIED

- B.O.C Kenya Ltd Ord 5.00 → Trading Symbol:BOC → ISIN CODE:KE0000000042
- British American Tobacco Kenya Ltd Ord 10.00 → Trading Symbol:BAT → ISIN CODE:KE0000000075
- Carbacid Investments Ltd Ord 5.00 → Trading Symbol:CARB → ISIN CODE:KE0000000117
- East African Breweries Ltd Ord 2.00 → Trading Symbol:EABL → ISIN CODE:KE0000000216
- Mumias Sugar Co. Ltd Ord 2.00 → Trading Symbol:MSC → ISIN CODE:KE0000000372
- Unga Group Ltd Ord 5.00 → Trading Symbol:UNGA → ISIN CODE:KE0000000497
- Eveready East Africa Ltd Ord.1.00 → Trading Symbol:EVRD → ISIN CODE:KE0000000588
- AFRICA MEGA AGRICORP PLC 5.00 → Trading Symbol: AMAC → ISIN CODE:KE0000000331
- Flame Tree Group Holdings Ltd Ord 0.825 → Trading Symbol:FTGH → ISIN CODE:KE4000001323
- Shri Krishana Overseas (SKL) → Trading Symbol:SKL.O0000 → ISIN: KE9900001216

TELECOMMUNICATION AND TECHNOLOGY

- Safaricom PLC Ord 0.05 → Trading Symbol:SCOM → ISIN CODE:KE1000001402

REAL ESTATE INVESTMENT TRUST

- Laptrust Imara I-REIT → Trading Symbol:LAPR → ISIN CODE:KE9100008870

EXCHANGE TRADED FUND

- New Gold Issuer (RP) Ltd → Trading Symbol:GLD → ISIN CODE:ZAE000060067
- Satrix MSCI World Feeder ETF → Trading Symbol: SMWF.E0000 → ISIN CODE:ZAE000246104



# Nairobi Securities Exchange (NSE) ESG Disclosures Guidance Manual – November 2021  
## Chatbot Fact Sheet (Exact Quotations – Ready for Copy & Paste)

**Document Title**  
"Nairobi Securities Exchange ESG Disclosures Guidance Manual"  
**Publication Date**  
NOVEMBER 2021  
**Total Pages**  
56  

**Acknowledgement (Page 2)**  
"The Nairobi Securities Exchange would like to thank the Global Reporting Initiative (GRI), the Swedish International Development Cooperation Agency (SIDA), the African Securities Exchanges Association (ASEA) and Seven Levers LLP, for their technical and financial support in the development of this ESG Disclosures Guidance Manual."

**Foreword – Geoffrey O. Odundo, Chief Executive Officer, Nairobi Securities Exchange (Page 3)**  
"By issuing these guidelines, the NSE aims at improving and standardizing ESG information reported by listed companies in Kenya. These guidelines provide a granular, tactical approach to ESG reporting that meets international standards on ESG reporting. Further guidance has been provided on how listed companies can integrate ESG considerations into their organisations, helping capture significant opportunities for stakeholders while managing critical business risks."

"Consistent application of these guidelines will help improve the capital markets in Kenya by providing information that investors are now demanding to facilitate decision making and capital allocation. This is a key objective of the capital markets."

**Foreword – Peter Paul van de Wijs, Chief External Affairs Officer, Global Reporting Initiative (GRI) (Page 4)**  
"Transparency on the impacts of a business is essential for continuous improvement as well as for stakeholder relationships. Without transparency, there is no trust – and without trust, markets do not function efficiently, and institutions lose their legitimacy."

"We therefore commend the Nairobi Securities Exchange (NSE) for producing this new ESG guide, which will help Kenyan companies to be accountable for their impacts while increasing their competitiveness in the global marketplace."

**Recommended Reporting Framework (Page 8)**  
"To help reduce uncertainties on which framework or standards to apply, this manual recommends the adoption of the GRI Standards as the common framework for ESG reporting by listed companies in Kenya."

"The GRI Standards are the most widely used framework for sustainability reporting."

**Definition – ESG Reporting (Page 7)**  
"An organisation’s practice of reporting publicly on its economic, environmental, and/or social impacts, and hence its contributions – positive or negative – towards the goal of sustainable development. Also commonly referred to as sustainability reporting or non-financial reporting."

**Definition – Materiality Assessment (Page 7)**  
"The process of prioritizing ESG topics for reporting and based on the assessed needs of different stakeholders."

**Top ESG Metrics Sought by Investors (IHS Markit) (Page 9)**  
• Presence of an overarching ESG policy.  
• Assignment of ESG management responsibility.  
• Corporate code of ethics  
• Presence of litigation.  
• People diversity.  
• Net employee composition.  
• Environmental policy.  
• Estimation of carbon footprint.  
• Data and cybersecurity incidents.  
• Health and safety events.

**Expected Benefits for Listed Companies (Page 9)**  
1. Transparency in ESG disclosures helps in building integrity and trust in the capital markets thus enhancing competitiveness to attract investment to the capital markets.  
2. Investors can assess and preferentially invest in issuers that demonstrate better ESG linked financial performance, resulting in more efficient capital allocation.  
3. Organisations that demonstrate responsible investment practices can access new sources of capital from sustainability conscious investors such as Development Finance Institutions (DFIs) and Private Equity firms.  
4. A wholistic view of corporate value facilitates product innovation by enabling consideration and management of the embodied environmental and social impacts of products and services.  
5. Measuring and reporting ESG performance enables organisations embed circularity in their operating models and achieve operational efficiencies by optimizing energy and raw costs in production.

**Key Steps in the ESG Reporting Process (Pages 13-17)**  
Step 1: Governance over ESG integration and reporting  
Step 2: Situational analysis and stakeholder engagement  
Step 3: Materiality analysis  
Step 4: Value creation  
Step 5: Content development  
Step 6: Assurance and internal controls

**Role of the Board (Page 14)**  
"What is the role of the Board?  
The Board has ultimate responsibility for ESG integration and reporting."

**Role of the CEO (Page 15)**  
"What is the role the Chief Executive Officer?  
The CEO is responsible for the day-to-day management of ESG issues and reporting."

**Mandatory Use of GRI Standards (Page 27)**  
"Adopt reporting standard  
This manual recommends the adoption of the GRI Standards as the common framework for ESG reporting by listed companies in Kenya."

**Annex 5 Criteria (Page 48)**  
"Criteria to claim an ESG report has been prepared in accordance with the GRI Standards"

**Future Plan (Page 8)**  
"A responsible investment index is planned by the Nairobi Securities Exchange (NSE) in future."

**Primary References (Page 10)**  
• Global Reporting Initiative (GRI) Sustainability Reporting Standards, 2018  
• The Code of Corporate Governance Practices for Issuers of Securities to the Public, 2015 (CMA Code)  
• United Nations Sustainable Development Goals (SDGs)

**Supported By**  
Global Reporting Initiative (GRI)  
Swedish International Development Cooperation Agency (SIDA)  
African Securities Exchanges Association (ASEA)  
Seven Levers LLP


### NSE Main Investment Market Segment (MIMS) Fact Sheet

This is the premium Board for large and well established companies in Kenya and the region. It hosts leading enterprises across a wide range of sectors in the economy.

Requirements to List on the Main Market Investment Segment

| Standard | Requirement |
| --- | --- |
| Incorporation status | Issuer to be a company limited by shares and registered under the companies act. |
| Share capital | Have a minimum authorised, issued and paid up ordinary share capital of Kshs. 50 Million. |
| Net assets | Net assets immediately before the public offering or listing of shares should not be less than Kshs. 100m. |
| Share transferability | Shares to be listed shall be freely transferable and not subject to any restrictions on marketability or any preemptive rights. |
| Competence of directors and management | Directors and Management must be ethical, not bankrupt, Not have any criminal proceedings. Must have suitable senior management with relevant experience for at least one year prior. Must not be in breach with any loan covenant particularly on debt capacity. |
| Availability and Reliability of Financial Records | The issuer shall have audited financial statements complying with International Financial Reporting Standards (IFRS) for an accounting period ending on a date not more than four months prior to the proposed date of the offer or listing for issuers whose securities are not listed at the securities exchange, and six months for issuers whose securities are listed at the securities exchange. |
| Dividend policy | Issuer must have a clear future dividend policy. |
| Track Record, Profitability and Future Prospects | The issuer must have declared profits after tax attributable to shareholders in at least three of the last five completed accounting periods to the date of the offer |
| Solvency and Adequacy of Working Capital | The issuer should not be insolvent The issuer should have adequate working capital. |
| Share and ownership structure | At least 25% of share to be held by not less than 1000 shareholders excluding employees of the issuer. |
| Certificate of comfort | In case of listing outside the Kenyan jurisdiction obtain a cert of no objection from the foreign securities exchange and regulator. |
| Listed shares to be immobilized | N/A |
| Nominated advisor | N/A |

### NSE Privacy Policy Fact Sheet (Last updated: May 12, 2025)

Introduction

This Privacy Policy outlines how we collect, use, store, and share your personal data when you register for or interact with our services. It reflects our commitment to data protection under local and international laws where applicable.

By providing your personal data, you consent to its processing as described in this policy.

1. Definitions

- Account means a unique record created for you in our registration forms or system.
- Company or NSE (referred to as either “the Company”, “We”, “Us”, “Our” or “NSE” in this Agreement) refers to the Nairobi Securities Exchange Plc.
- Country refers to Kenya (or your applicable jurisdiction).
- Device means any device that can be used to provide personal information, such as a computer, mobile phone, or tablet.
- Personal Data means any information relating to an identified or identifiable individual.
- Service refers to all the products and functions the Company offers, including websites, apps, and any in-person or physical services like events.
- Service Provider means any natural or legal person who processes data on behalf of the Company.
- Third-party refers to any platform not owned by the Company that may be used to engage with our services.
- Usage Data refers to data collected automatically through interaction with our platforms or communications.
- You means the individual providing information or the organization on behalf of which such individual is providing information.

2. Legal Basis for Processing

We process your personal data under one or more of the following legal bases:

0. Consent – for marketing communications and optional services.
1. Contractual necessity – to register and facilitate your participation in events or utilization of our services.
2. Legal obligation – where required by law or regulatory frameworks.
3. Legitimate interest – such as managing and improving events/services, security, and engagement analytics.

3. Types of Data Collected

We may collect the following types of personal data:

i. Full name

ii. Email address

iii. Phone number

iv. Gender

v. Personal Identification/Passport number/Tax Identification

vi. Age

vii. Address, city, county, and country

viii. Organization and job title

ix. Education and qualifications (if necessary)

x. Dietary or accessibility requirements

xi. Emergency contact (if necessary)

xii. Event participation history

xiii. Preferences for communication or future contact

xiv. Feedback, survey responses, and evaluations

We do not collect sensitive personal data unless it is strictly necessary for event or service delivery and provided with explicit consent.

4. Children’s Privacy

We do not knowingly collect personal information from children under the age of 18 without verified parental or guardian consent. If we discover that personal data from a child has been collected without appropriate consent, we will delete it promptly. For youth-focused events or services, only essential information will be collected, and only with verified guardian authorization.

5. Use of Your Personal Data

We use your personal data for the following purposes:

i. To register and manage your participation in events or utilization of our services.

ii. To communicate updates, confirmations, and reminders

iii. To provide personalized experiences and logistical support

iv. To analyze service utilization, attendance, feedback, and engagement metrics

v. To send marketing communications for future events (where you have consented)

vi. To comply with applicable laws and regulations

vii. To process applications made by you or your organization.

6. Marketing Communications

With your explicit consent, we may use your contact information to send promotional materials, newsletters, event invitations, or other communications we believe may interest you. You can opt out at any time by contacting us directly or utilizing opt out mechanisms that may be provided with the service.

7. Data Retention

We determine the length of data retention on a case-by-case basis, depending on factors such as the nature of the information, the purpose for its collection and processing, and any legal or operational retention requirements. Once your data is no longer necessary for these purposes, it will be securely deleted or anonymized.

8. Cross-Border Data Transfers

When personal data is transferred outside Kenya, we comply with the Data Protection Act, 2019, and ensure transfers are based on one or more of the following:

• Transfers to countries recognized as having adequate data protection.

• Use of standard contractual clauses, data transfer agreements, or other legally recognized safeguards.

• Binding corporate rules for intra-group transfers.

• Your explicit consent, where required.

We recognize that recipient countries may have different and potentially less stringent data privacy laws. Where necessary, we impose contractual obligations and require adherence to international codes of conduct to ensure your data is adequately protected.

9. Sharing of Personal Data

We may share your data with:

• Service Providers involved in service provision, event logistics, communications, and data processing

• Event Partners or Sponsors (only where relevant and with your consent)

• Regulators or legal authorities, when required by law or to protect rights

We do not sell or rent your personal data to third parties.

10. Security of Your Personal Data

We have procedures in place to respond to suspected personal data breaches and will notify you and any applicable regulators of a breach where we are legally required to do so. Access to your data is restricted to employees, agents, contractors, and third parties who have a business need to know. They will process your data only on our instructions and are subject to a duty of confidentiality.

11. Your Rights

Under the Kenya Data Protection Act and GDPR, you have the right to:

i. Access the personal data we hold about you

ii. Correct inaccurate or outdated data

iii. Request deletion of your data (right to be forgotten)

iv. Object to processing, especially for direct marketing

v. Withdraw consent at any time without affecting prior lawful processing

vi. Restrict processing under certain conditions

vii. Request portability of your data (where applicable)

viii. Lodge a complaint with the Office of the Data Protection Commissioner (ODPC) or other supervisory authority

To exercise these rights, please contact us using the details provided below.

12. Automated Decision-Making

Should we ever engage in automated decision-making or profiling that produces legal or similarly significant effects, we will provide you with clear information about the logic involved and the potential consequences, as well as your rights relating to such processing.

13. Cloud Computing and Hosting

Your personal data may be held or processed in the cloud, in data centers or systems operated by approved third-party cloud providers located within or outside Kenya. All cloud providers engaged by NSE are required to comply with the Kenya Data Protection Act, GDPR (where applicable), and relevant international standards such as ISO/IEC 27001. We conduct due diligence and require contractual commitments (including SLAs) to maintain confidentiality, integrity, and availability of your data, including clarity on server location and ongoing monitoring of provider compliance.

14. External Links Disclaimer

Our website or services may, from time to time, contain links to external sites operated by partner networks, advertisers, or affiliates. Please note that these sites have their own privacy policies, and NSE does not accept responsibility or liability for their practices. We recommend reviewing their privacy statements before submitting any personal data.

15. Changes to This Privacy Policy

We reserve the right to update this policy from time to time to reflect changes in law, technology, or our practices. When we do, we will update the “Last Updated” date and provide appropriate notice for material changes.

Contact Us

Email: dataprotectionoffice@nse.co.ke

### NSE International Securities Identification Number (ISIN) Fact Sheet

An international securities identification number (ISIN) is a 12-digit code that is used to uniquely identify a security’s issue e.g. shares, bonds, etc. This is currently the main method of securities identification worldwide.

The Nairobi Securities Exchange PLC (NSE) is the recognized numbering authority for issuing ISINs for Kenya, as authorized by the Association of National Numbering Agencies (ANNA).

You can read the https://www.anna-web.org/ to find out more.

You can download ISIN files for Equities, Bonds and Derivatives.

Disclaimers

All data and information provided by the NSE, except as otherwise indicated, is proprietary to the NSE. You may not copy, reproduce, modify, reformat, download, store, distribute, publish or transmit any data and information, except for your personal use. For the avoidance of doubt, you may not develop or create any product that uses, is based on, or is developed in connection with any of the data and information available on this site. You are not permitted (except where you have been given express written permission by the NSE) to use the data and information for commercial gain.

###  NSE Our Story / History Fact Sheet

History of NSE
1920s – 1953: Dealing in shares commenced with trading taking place on a gentleman’s agreement with no physical trading floor. London Stock Exchange (LSE) officials accepted to recognize the setting up of the Nairobi Stock Exchange as an overseas stock exchange (1953).
1954 – 1962: The Nairobi Stock Exchange (NSE) was registered under the Societies Act (1954) as a voluntary association of stockbrokers and charged with the responsibility of developing the securities market and regulating trading activities. Business was transacted by telephone and prices determined through negotiation.
1963 – 1970: The Government adopted a new policy with the primary goal of transferring economic and social control to citizens. By 1968, the number of listed public sector securities was 66 of which 45% were for Government of Kenya, 23% Government of Tanzania and 11% Government of Uganda. During this period, the NSE operated as a regional market in East Africa where a number of the listed industrial shares and public sector securities included issues by the Governments of Tanzania and Uganda (the East African Community).

However, with the changing political regimes among East African Community members, various decisions taken affected the free movement of capital which ultimately led to the delisting of companies domiciled in Uganda and Tanzania from the Nairobi Stock Exchange.

1975: When the EAC finally collapsed in 1975, the Government of Uganda compulsorily nationalized companies which were either quoted or subsidiaries of listed companies.

1988: The first privatization through the NSE, through the successful sale of a 20% Government stake in Kenya Commercial Bank. The sale left the Government of Kenya and affiliated institutions retaining 80% ownership of the bank.

1990: The CMA was constituted in January 1990 through the Capital Markets Authority Act (Cap 495A) and inaugurated in March 1990. The main purpose of setting up the CMA was to have a body specifically charged with the responsibility of promoting and facilitating the development of an orderly and efficient capital market in Kenya.

1991: NSE was registered as a private company limited by shares. Share trading moved from being conducted over a cup of tea, to the floor based open outcry system, located at IPS Building, Kimathi Street, Nairobi.

1993: The CMA increased the initial paid up capital for stockbrokers from Kshs.100, 000 to Kshs.5.0 million while that for investment advisors was set at Kshs.1.0 million.

1994: With the 1994 CMA Act (Amendments), it became mandatory that a securities exchange approved by the CMA be a company limited by guarantee. The number of stockbrokers increased by a further seven.

On February 18, 1994, the NSE 20-Share Index reached a record high of 5,030 points. The NSE was rated by the International Finance Corporation (IFC) as the best performing market in the world with a return of 179% in dollar terms.

The NSE also moved to more spacious premises at the Nation Centre in July 1994, setting up a computerized delivery and settlement system – DASS.

1995: An additional eight stockbrokers were licensed in June 1995 and with the suspension of one stockbroker, the total number of stockbrokers was twenty.

The CMA established the Investor Compensation Fund whose purpose was to compensate investors for financial losses arising from the failure of a licensed broker or dealer to meet their contractual obligations.

1996: Privatization of Kenya Airways where more than 110, 000 shareholders acquired stake in the airline and the Government of Kenya reduced its stake from 74% to 26%. The Kenya Airways Privatization team was awarded the World Bank Award for Excellence for 1996, for being a model success story in the divestiture of state-owned enterprises.

1997: With the objective of developing a code of conduct, promoting professionalism, and establishing examinable courses for its members as well as facilitate liaison with the CMA and the NSE, the members of the NSE formed the Association of Kenya Stock brokers (AKS).

1998: The CMA published new guidelines on the disclosure standards by listed companies. The disclosure requirements were meant for both public offerings of securities as well as continued reporting obligations, among others.

January 1999: The CMA issued guidelines to promote good corporate governance practices by listed companies through the constitution of audit committees.

March 23, 1999: The Central Depository and Settlement Corporation Limited (CDSC) was incorporated under the Companies Act (Cap 486).

November 30, 1999: The East African Community Treaty was signed in Arusha, Tanzania between five countries, Burundi, Kenya, Rwanda, Uganda and The United Republic of Tanzania.

2000: Five core shareholders of the CDSC signed an agreement and paid up some share capital.

2001: The market at the NSE was split into the Main Investment Market Segment (MIMS), Alternate Investment Market Segment (AIMS) and the Fixed Income Securities Market Segment (FISMS).

The EAC Secretariat formally convened the first meeting of the new Capital Markets Development Committee in Dar es Salaam, Tanzania.

2003: The Central Depositories Act 2000 was operationalized in June 2003.

2004: Following the successful signing of an MOU between the Dar-es-Salaam Stock Exchange, the Uganda Securities Exchange and the Nairobi Securities Exchange, the East African Securities Exchanges Association was formed.

November 10, 2004: The central despository system was commissioned. For the first time in Kenya’s history, the process of clearing and settlement of shares traded in Kenya’s capital markets was automated.

September 11, 2006: The NSE implemented live trading on its own automated trading systems trading equities. The ATS also had the capability of trading immobilized corporate bonds and treasury bonds. The Exchange’s trading hours were increased from two hours (10:00 am – 12:00 pm) to three hours (10:00 am – 1:00 pm).

December 17, 2007: The NSE implemented its Wide Area Network (WAN) platform. With the onset of remote trading, brokers and investment banks no longer required a physical presence on the trading floor since they would be able to trade through terminals in their offices linked to the NSE trading engine.

February 1, 2008: The Nairobi Stock Exchange (NSE) announced the extension of trading hours at the bourse. Trading would commence from 9.00am and close at 3.00pm each working day.

February 25, 2008: In order to provide investors with a comprehensive measure of the performance of the stock market, the Nairobi Stock Exchange introduced the NSE All-Share Index (NASI).

April 2008: The NSE launched the first edition of the NSE Smart Youth Investment Challenge to promote stock market investments among Kenyan youth.

June 9 2008: The immobilized shares of Safaricom Ltd., commenced trading on the NSE after the trading session was opened in a colorful ceremony presided by H.E. President Mwai Kibaki. The Safaricom IPO increased the number of shares listed on the bourse to over 55.0 billion shares, from the previous 15.0 billion.

2009: The Exchange launched its Complaints Handling Unit (CHU) in a bid to make it easier for investors and the general public to forward any queries and access prompt feedback.

December 7, 2009: The NSE marked the first day of automated trading in government bonds through the Automated Trading System and uploaded all government bonds on the System.

2011: July 4, 2011: The equity settlement cycle moved from the previous T+4 settlement cycle to the T+3 settlement cycle.

July 6, 2011: The Nairobi Stock Exchange Limited changed its name to the Nairobi Securities Exchange Limited. The change of name was a reflection of the 2010 – 2014 strategic plan of the Nairobi Stock Exchange to evolve into a full service securities exchange which supports trading, clearing and settlement of equities, debt, derivatives and other associated instruments.

August 1, 2011: The business segments under which our listed companies are placed were reclassified. Equities were now under ten (10) industry sectors. Debt securities including preference shares were under three (3) categories.

November 8, 2011: The NSE together with FTSE International launched the FTSE NSE Kenya 15 and FTSE NSE Kenya 25 Indices.

March 21, 2012: The Nairobi Securities Exchange became a member of the Financial Information Services Division (FISD) of the Software and Information Industry Association (SIIA).

August 8, 2012: The Nairobi Securities Exchange entered into a Memorandum of Understanding with the Somalia Stock Exchange Investment Corporation (SSE), regarding the possibility of co-operating to establish a securities exchange business involving the trading, settlement, delivery of listed securities and other stockbrokerage activities.

September 5, 2012: The NSE Broker Back Office commenced operations with a system capable of facilitating internet trading improving the integrity of the Exchange trading systems.

October 3, 2012: NSE together with FTSE International launched the FTSE NSE Kenyan Shilling Government Bond Index. This was the first instrument of its kind in Eastern Africa and gave investors the opportunity to access current information and provided a reliable indication of the Kenyan Government Bond market’s performance.

December 14, 2012: UMEME Holdings Limited cross listed its 1,623,878,005 shares on the Main Investment Market Segment (MIMS) of the Nairobi Stock Exchange. It was the first inward cross listing of an East African company on the NSE since the incorporation of the East African Securities Exchanges Association (EASEA) on May 15 2009.

January 22, 2013: The Growth Enterprises Market Segment (GEMS) giving small and medium enterprises a great opportunity to access the capital markets was launched by Mr. Mugo Kibati, Director-General of Kenya Vision’s 2030 Delivery Board.

February 25 2013: Centum Investment Company became the first company in East Africa to list an equity linked note, when its Kshs. 4.19 Billion Note Issue commenced trading on the Nairobi Stock Exchange (NSE) Fixed Income Securities Market Segment (FISMS).

February 28, 2013: The Board of Association of Futures Market (AFM), admitted the Nairobi Securities Exchange (NSE) as an associate member of the Association. The Association promotes and encourages the establishment of new derivatives and related markets.

June 3, 2013: The Nairobi Securities Exchange moved to its new residence outside the central business district located on 55 Westlands Road – The Exchange.

June 25, 2013: The first reverse takeover transaction in East Africa was completed when I&M Holdings began trading on the NSE. I&M Holdings used the reverse takeover of City Trust which was listed on AIMS to migrate the new firm to MIMS.

June 26, 2013: Moody’s Investors Service and the Nairobi Securities Exchange hosted the inaugural East Africa Credit Risk Conference in Nairobi.

July 15, 2013: Home Afrika, made history by being the first company to list by introduction on the Growth Enterprise Market Segment (GEMS).

August 8, 2013: The Nairobi Securities Exchange (NSE) and the Shanghai Stock Exchange (SSE) entered into a Memorandum of Understanding (MoU).

August 19, 2013: The Board of Directors of the Nairobi Securities Exchange admitted CBA Capital Ltd., and Equity Investment Bank Ltd., as trading participants.

September 24, 2013: As a result of its initiatives to increase company listings and diversify asset classes, the panel of distinguished judges for the 2013 Africa investor Index Series Awards, ranked the Nairobi Securities Exchange (NSE), the winner of the Most Innovative African Stock Exchange category.

April 7, 2014: The 60th Anniversary Celebrations of the Nairobi Securities Exchange kicked off with the Launch of the new NSE Brand and official opening of the Exchange. Our brand migration was informed by developments we are undergoing at the bourse.

June 27, 2014: The Nairobi Securities Exchange (NSE) received formal approval from the Capital Markets Authority (CMA) to operate as a demutualized entity. This is after the approval of the NSE’s final application which met the regulator’s requirements as stipulated in Section 5(3) of the Capital Markets (Demutualization of the Nairobi Stock Exchange) Regulations 2012.

On June 27, 2014: The Nairobi Securities Exchange (NSE) received formal approval from the Capital Markets Authority (CMA) to offer its shares to the public through an Initial Public Offer (IPO) and subsequently Self-list its shares on the Main Investment Market Segment (MIMS) of the NSE.

July 23, 2014: The Nairobi Securities Exchange Limited officially launched its Initial Public Offering (IPO) seeking to raise Kshs.627,000,000.00 by selling up to 66,000,000 new shares at a price of Ksh 9.50 per share. The minimum number of shares available for purchase were 500.

September 9, 2014: The Nairobi Securities Exchange listed its 194,625,000 issued and fully paid up shares on the Main Investment Market Segment (MIMS) under a new sector – Investment Services of the bourse after a successful Initial Public Offering which sought to raise Kshs.627 million by selling 66 million new shares at a price of Kshs. 9.50 per share. 17,859 investors applied for 504,189,700 new shares worth Kshs. 4.789 billion; a subscription rate of 763.9%, garnering an oversubscription of 663.92%. Following its self-listing the Exchange becomes the second African Exchange after the Johannesburg Stock Exchange to be listed.

On September 16, 2014: The Nairobi Securities Exchange was added as a constituent of the auspicious FTSE Mondo Visione Exchanges Index, the first Index in the world to focus on listed exchanges and other trading venues.

September 26, 2014: The Nairobi Securities Exchange launched a new system for trading corporate bonds and Government of Kenya Treasury Bonds allowing on-line trading of debt securities and is integrated with the settlement system at the Central Bank of Kenya. The system is more efficient, scalable and flexible, and can support trading in bonds that have been issued in foreign currencies.

October 24, 2014: The Nairobi Securities Exchange (NSE) and Korea Exchange (KRX) signed a memorandum of Understanding (MOU) in Korea, marking the beginning of collaboration between the Kenya and Korea Capital Markets

### NSE Training Fact Sheet

About NSE Training

Nairobi Securities Exchange PLC offers a wide range of premium trainings by the leading market experts.Our customized trainings equip both businesses and individuals with prerequisite skills and knowledge on capital and financial markets through high quality, cost effective and accessible learning. The highly supportive and interactive learning platform incorporates technology that enhances skills, capacity building and offers certification in the Kenyan Market.

Our Courses

Our courses are designed to provide individuals and corporates a better understanding of the various aspects and products of the capital and finacial markets. The courses are geared towards bridging knowledge gap and offering financial inclusion to empower capital market professionals, investors and the general public.

We have partnered with other leading organizations to deliver a full range of financial education courses that create impact and sustainable growth through powerful learning. The courses offers participants Continous Professional Development that allows learning in a structured and practical format and ascertain skills and knowledge in various specialities as well as maintain competence through various professional body memberships.

The courses combine practical application with academic excellence through interactive face to face classroom teaching by our qualified professional instructors and e-learning innovation using collaborative learning methodology that allows participants to learn by sharing ideas which forms a core part of all our trainings.

### NSE Exchange Traded Funds (ETFs) Fact Sheet

An ETF is defined as a listed investment product, which tracks the performance of a particular index (e.g. NSE 20, NSE 25) or “basket” of shares, bonds, money market instruments or a single commodity. These are known as underlying securities or assets. ETF are traded on an exchange just like an ordinary share and the price of a particular ETF will be determined by the demand and supply of the ETF. An ETF can be a domestic or offshore product

A. Types of exchange- traded funds

- Index ETFS – Most ETFs are index funds that attempt to replicate the performance of a specific index. Indexes may be based on stocks, bonds, commodities, or currencies. An index fund seeks to track the performance of an index by holding in its portfolio either the contents of the index or a representative sample of the securities in the index.
- Bond ETFs – They invest in bonds. They thrive during economic recessions because investors pull their money out of the stock market and into bonds (for example, government treasury bonds or those issued by companies regarded as financially stable).
- Commodity ETFs (ETCs) – Invest in commodities such as precious metal, agricultural products and hydrocarbons.
- Stock ETFs –This was the first and most popular ETFs. This type of ETF owns funds from other stocks

B. Benefits of exchange traded funds

* Diversification – ETF give investors exposure to a wide variety of securities or assets, avoiding the risk of “putting all your eggs in one basket”.
* Regulation – ETF are well regulated by both the Nairobi Securities Exchange (NSE) and Capital Markets Authority (CMA). Investors therefore have added protection against unjust treatment.

  Cash flow distributions (dividends) – Even though owning an ETF does not give direct ownership of the underlying securities of the index being tracked, owners of an ETF are still eligible to receive dividends should the securities in the tracking index pay dividends.
* Liquidity – Buying or selling ETF can be done quickly and at a low cost at the NSE.
* Tax – ETF are exempt from Capital Gains Tax upon sale of the ETF.
* Hassle free investment – Investors can gain exposure to a wide variety of securities or assets without having to buy each of the underlying constituents individually, conducting extensive research, nor actively managing the underlying securities
* Transparency – ETFs disclose their holdings on a daily basis, thereby enabling investors to know exactly what stocks or underlying assets they hold, what the value is and to make more informed investment decisions
* Flexibility – Like an equity market, ETFs trade throughout market hours.

Parties involved in an ETF issuance
Market Maker -a market maker in Kenya’s ETF market shall play the role of creating liquidity through two-way price quotes in order to eradicate substantial price gaps and ensure a liquid market for all.
Manager – The Fund Manager acts on its behalf for the purpose of managing the ETF or undertake to manage the ETF.
Trustee -A trustee primary responsibility of protecting the interests of investors.
Current Issuances in the market

|  | Issuer | Name | Type of ETF | Listing Date |
|  | --- | --- | --- | --- |
|  | Absa Group | Absa Gold Backed | Commodity ETF | 2016 |



### 2. Single Stock Futures Product Specifications

**Fact Sheet: Single Stock Futures Contract Specifications (NSE Derivatives Market)**

Category of contract: Single Stock Future

Underlying financial instrument: Single stock listed on the NSE (E.g. Equity Group Holdings Plc. – EQTY)

System code: Jun19 EQTY

Contract months: Quarterly (March, June, September and December)

Expiry dates: Third Thursday of expiry month. (If the expiry date is a public holiday then the previous business day will be used.)

Expiry times: 15H00 Kenyan time

Listing program: Quarterly

Valuation method on expiry: Based on the volume weighted average price (VWAP) of the underlying instrument for liquid contracts, and the theoretical price (spot + cost of carry) for illiquid contracts.

Settlement methodology: Cash settlement

Contract size:
- For shares trading below KES 100: One contract equals 1,000 underlying shares.
- For shares trading above KES 100: One contract equals 100 underlying shares.

Minimum price movement (Tick Size):
Price Range          | Tick Size (KES)
--------------------|----------------
Below 100.00        | 0.01
≥ 100.00 < 500.00   | 0.05
≥ 500.00            | 0.25

Initial Margin requirements: As determined by the NSE Methodology.

Mark-to-market: Explicit daily. Based on the volume weighted average price (VWAP for liquid contracts, theoretical price for illiquid.

Market trading times: 09H30 to 15H00 Kenyan time

Market fees (as % of notional contract value):

Participant         | Percentage
--------------------|------------
NSE Clear           | 0.025%
Clearing Member     | 0.025%
Trading Member      | 0.10%
IPF Levy            | 0.01%
CMA Fee             | 0.01%
TOTAL               | 0.17%


### NSE Bond Index (NSE-BI) Ground Rules v1.2 – Comprehensive Fact Sheet
**Document Title**  
GROUND RULES FOR GENERATION OF NSE BOND INDEX (NSE-BI)  
V1.2  

Approved by: Trading Committee  

**Section 1: Introduction**  
The Index shall be generated and distributed under the name, ‘the NSE Bond Index’ (NSE-BI).  
The NSE-BI is a broad, comprehensive, market value weighted Index designed to measure the performance of the Kenyan Treasury bond market. Constituent bonds that are factored in the calculation of the NSE-BI are derived from the benchmark bonds issued by the Government.  
The benchmark bonds are 2,5,10,15,20,25 years to maturity. Any new bond issued by the government (on the run bonds) will be included into the index on the date of listing at the NSE.  

This manual outlines the procedures that guide the generation and management of the NSE-BI. It tackles issues concerning Bond eligibility criteria, index construction and maintenance methodology, determination of the index base date and dissemination of index results.

**Section 2: Eligibility Criteria (Verbatim)**  
A bond must meet all the following criteria on the rebalancing date to be Included in the index.  

➢ Issuer: The bond issuer will be the Government of Kenya though its fiscal agent the Central Bank of Kenya. (CBK) and listed at the Nairobi securities Exchange.  
➢ Denomination: The bond must be denominated in Kenya shillings (KES)  
➢ Minimum Par Amount: The amount outstanding, or Par Amount, is used to determine the weight of the bond in the Index. The bond must have a minimum Par Amount of Kes 5 billion to be eligible for inclusion. To remain in the Index, bonds must maintain a minimum Par amount greater than or equal to Kes 1 billion as of the next rebalancing date.  
➢ Minimum Term: As of the next Rebalancing date, the bond must have a minimum term to maturity greater than or equal to one year (364 days)  
➢ Benchmark bond: the bond to be included in the index must be a benchmark bond. Benchmark bonds are securities or debt instruments mainly issued for market development. They are characterized by high credit quality, least default risk, are highly liquid and reasonably large in volume. As a result, these bonds are actively traded in the secondary market, have stable prices/yields and are therefore most reliable the basis for deriving benchmark measures in this case the bond index. In Kenya benchmark bonds include bonds of 2, 5, 10, 15,20 and 25-year benchmark Maturities.

**Section 3: Index Construction & Maintenance Methodology (Verbatim Extracts)**  
The NSE-BI is a clean price market-value-weighted index. The prices used to generate the index are the volume weighted prices of the selected benchmark bonds as concluded and reported via the automated trading system (ATS). Where there are no trades, the Exchange shall call for firm two-way quotes from licensed trading participants.  

The Market value of a bond equals the adjusted amount outstanding, multiplied by the clean price, expressed as a percentage. Capitalization weighting effectively assumes that an investor "buys the market" and is therefore the most appropriate weighting scheme for a performance benchmark.  

Valuation: The securities that make up the NSE-BI are priced each end of week by the trading desk based on the executed trades as reported via the ATS. For purposes of index calculation, the volume weighted yields or quotes where applicable are used to generate the yields input for index calculation. If an index linked issue does not trade, the last traded clean price will be used for index generation. To improve the accuracy of the index, data that will be considered will only be of trades that are Kes 50M plus to avoid trades that distort the index.  

Settlement Conventions: Treasury bonds in Kenya accrue interest using an actual/364-day count convention. Accrued interest on bonds in the index is calculated assuming three-day settlement (T+3).  

Index Maintenance Rules:  
• All benchmark bonds are eligible for inclusion in the index on the rebalancing date.  
• Any Index Bond that fails to meet any one of the eligibility factors, or that will have a term to maturity and/or call date less than or equal to one year (364-day count) of the next Rebalancing Date, will be removed from the Index on that Rebalancing Date.  
• Par amounts of Index Bonds will be adjusted on the Rebalancing date to reflect any changes that have occurred since the previous Rebalancing Date, due to Partial calls, rediscounting etc.

Re-balancing: The Index is reviewed and monthly. The Index Committee, nevertheless, reserves the right to adjust the Index at any time that it believes appropriate balancing takes place on the last business day of each month. Additions, deletions, and other changes to the Index arising from the monthly rebalancing shall be published via the NSE website.

**Section 4: Index Base Date (Verbatim)**  
The NSE-BI base date is quoted at 1000 on September 8, 2023.

**Section 5: Dissemination of New Index Results (Verbatim)**  
The Nairobi Securities Exchange shall ensure that the following information on the indices is widely published:  
• Index values  
• List of constituents  
• Changes to constituents  
• Changes and amendments to the Ground Rules  
• Details of any recalculations or calculation amendments.  

Constituent Bond data together with statistics on the indices are available from the NSE Website. The weekly index points will be published on the weekly Implied Yield Pricelist and the NSE Website before the close of business.

**Section 6: Index Governance (Verbatim)**  
The NSE Trading & Technology Committee, a Committee of the NSE Board maintains the Index. The Committee oversees the day-to-day management of the Index, including the monthly rebalancing, determinations of intra-rebalancing changes to the Index, and maintenance and inclusion policies, including additions or deletions of bonds and other matters affecting the maintenance and calculation of the Index.  

In fulfilling its responsibilities, the Committee has full and complete discretion to (i) amend, apply, or exempt the application of Index rules and policies as circumstances may require and (ii) add, remove, or by-pass any bond in determining the composition of the Index.

**Section 8: Current NSE-BI Constituent Bonds (As of Latest Update – Verbatim Table)**  

No. | Constituents    | ISIN           | Outstanding Amount Shs'M.
----|-----------------|----------------|----------------------------
1.  | FXD1/2023/002  | KE8000006109  | 94,724.39
2.  | FXD1/2024/003  | KE8000006323  | 91,555.15
3.  | FXD1/2023/005  | KE8000005986  | 144,534.30
4.  | FXD1/2022/010  | KE7000009436  | 80,901.70
5.  | FXD1/2012/020  | KE4000003949  | 130,805.92
6.  | FXD1/2024/010  | KE8000006547  | 124,539.40
7.  | FXD1/2010/025  | KE4000003089  | 20,192.50
8.  | FXD2/2018/020  | KE5000008655  | 89,198.60
9.  | FXD1/2021/025  | KE7000003652  | 90,490.00
10. | FXD1/2022/025  | KE8000005093  | 103,141.56




### 6. NSE Corporate Social Responsibility (CSR) – Charity Trading Day

**Fact Sheet: NSE CSR Initiatives (Charity Trading Day)**

The NSE Charity Trading Day is an annual event that brings together celebrities, capital market participants, sponsors, and beneficiaries to network and have fun while encouraging investors to make a trade on that day in support of the event’s cause.

During the Charity Trading Day, celebrities and other participants make calls to clients on behalf of dealers, with all of the day’s trading revenues being donated towards Charity.

Focus Areas by Year:
- 2015–2017 → Wildlife conservation (e.g. Borana Ranch Conservancy, Nature Kenya)
- 2018 → Education (aligned to UN SDG 4 – inclusive and equitable quality education)
- 2019 → Health (aligned to UN SDG 3 – National Cancer Institute of Kenya data registry system)
- 2021 → Support for SMEs affected by COVID-19 through the KEPSA revolving fund

Beneficiaries Supported:
- Borana Ranch Conservancy
- We The Change Foundation
- Genevive Audrey Foundation
- SOS Children’s Home
- Joy Children’s Home
- Nature Kenya
- The National Cancer Institute of Kenya
- KEPSA

The Charity Trading Day has made a significant difference towards wildlife conservation, promotion of education and other humanitarian causes.

If your organization is aligned to our cause and wishes to be a beneficiary of the Charity Trading Day please contact info@nse.co.ke.

Thank you to all our sponsors for your generous contributions and continued support.


### 3. NSE Derivatives Rules (July 2017) – Comprehensive Fact Sheet

**Document Overview**  
NSE Derivatives Rules  
July 2017  

The purpose of these Derivatives Rules and directives is to achieve the objects of the NSE as set out in its Memorandum and Articles of Association by providing the procedures necessary to establish and regulate a fair and efficient derivatives market and to ensure that the business of the NSE is carried out in an orderly manner and with due regard to the objects of the Capital Markets Act.

**Key Provisions – Section 1: Derivatives Rules**  
- The overall management and control of the derivatives market shall be exercised by the Board of Directors of the NSE (“the Board”).  
- The Derivatives Rules and directives are binding on members, officers and their employees AND on any person utilising the services of a member or who concludes a transaction with a member.  
- Every transaction in derivative securities entered into by a trading member must be concluded on the specific condition that the transaction is entered into subject to the provisions of the Act, these Derivatives Rules and the directives.  
- Subject to the provisions of the Act, the NSE and the clearing house shall not be liable for any loss or damage resulting from negligence (except gross negligence), system malfunctions, or any other cause.  
- A Settlement Guarantee Fund and Investor Protection Fund are established with mandatory contributions from members.

**Key Definitions (Selected – Exact Wording)**  
“Act” means the Capital Markets Act (Chapter 485A of the Laws of Kenya)  
“additional margin” means the margin paid to a clearing member over and above that required by the clearing house  
“clearing member” means a sub-category of market participant of the NSE, registered to perform clearing in the derivatives market  
“client” means any person who uses the services of a market participant  
“default” means a default by a client or member as contemplated in rule 9  
“futures contract” means a contract in terms of which the seller is obliged to deliver or to pay an amount if the price/value changes  
“initial margin” means the amount determined by the clearing house as the initial deposit  
“mark-to-market” means the revaluation of open positions to current market prices  
“trading member” means a sub-category of market participant of the NSE, registered to perform trading in the derivatives market  
“variation margin” means the amount calculated daily to settle profits/losses on open positions

**Key Membership Categories**  
- Trading Member (non-clearing)  
- Clearing Member  
- Trading & Clearing Member  

**Key Obligations**  
- Members must maintain minimum capital requirements, fidelity insurance, and professional indemnity cover.  
- Members must segregate client funds and positions.  
- Daily mark-to-market and margin calls are mandatory.  
- Position limits and large exposure reporting apply.  
- Default waterfall: client margin → member margin → Settlement Guarantee Fund → IPF.

### 4. Policy Guidance Note on Green Bonds (January 2019) – Comprehensive Fact Sheet

**Document Overview**  
POLICY GUIDANCE NOTE ON GREEN BONDS  

This Policy Guidance Note (PGN) is a guide on the operational regulatory environment on Green Bonds in Kenya.  

January 2019  
Capital Markets Authority

**Key Definitions (Exact Wording)**  
“Green Bond” means a fixed income instrument, either unlisted or listed on a securities exchange, approved by the Authority, whose proceeds are used to finance or refinance new or existing projects that generate climate or other environmental benefits that conform to green guidelines and standards.  

“eligible projects” means project categories that contribute to environmental objectives including but not limited to climate change mitigation, climate change adaptation, natural resource conservation, biodiversity conservation, and pollution prevention and control.  

“greenwashing” means the superficial or insincere display of concern for the environment including mislabelling of a bond as green or overstatement of environmental benefits.

**Core Requirements for Green Bond Issuance & Listing**  
- Issuer must appoint an Independent Verifier (pre-issuance review and confirmation of green status).  
- The Information Memorandum must contain a statement/report from the Independent Verifier confirming alignment with green principles.  
- Issuer must develop a Green Bond Framework covering: Use of Proceeds, Project Evaluation & Selection, Management of Proceeds, Reporting.  
- Annual Green Bond Report required (allocation of proceeds + impact metrics), reviewed by Independent Verifier.  
- Proceeds can only be used for eligible green projects; tracking is mandatory.  
- Unallocated proceeds must be disclosed and temporarily placed in cash, cash equivalents or other green investments.  
- Breach of green requirements → Authority may direct removal of “green” label, impose sanctions, or suspend trading (for listed bonds).  

**Eligible Independent Verifiers**  
Must be CMA-approved entities with demonstrated expertise in environmental issues and green finance (e.g., CICERO, Sustainalytics, or local equivalents approved by CMA).  

**Continuous Obligations**  
- Annual reporting on allocation and impact until full allocation.  
- Post-issuance verification recommended (at least once).  
- Immediate disclosure if proceeds are reallocated away from green projects.


### NSE Equity Securities Trading Rules – Comprehensive Fact Sheet (Verbatim Key Extracts)

**Document Title**  
NAIROBI SECURITIES EXCHANGE TRADING RULES FOR EQUITY SECURITIES  

**Core Application (Rule 3)**  
These Rules shall apply to all Trading Participants, their authorised representatives, and all transactions in equity securities executed through the Automated Trading System (ATS) of the Nairobi Securities Exchange.

**Key Definitions (Exact from Section 2)**  
“Auction Call Session” means the session during which orders are entered but not matched until the opening price is determined.  
“Board Lot” or “Board Lot” means a standard trading unit of one hundred (100) equity securities.  
“Odd Lot” means a quantity of equity securities less than one Board Lot.  
“Reference Price” means the Volume Weighted Average Price (VWAP) of an equity security calculated from all transactions executed during the entire trading session of the previous business day.  
“Trading Participant” means a person licensed by the Capital Markets Authority as a stockbroker or dealer and authorised by the Exchange to trade on the ATS.  
“Manifest Error” means a transaction executed at a price that deviates significantly from the prevailing market price due to system malfunction, human error, or erroneous order entry.

**Trading Sessions & Hours (Exact from Rule 6.1 & 7)**  
Trading shall be conducted Monday to Friday (except public holidays) as follows:

Session                  | Time                  | Activity
-------------------------|-----------------------|-------------------------------------------------
Pre-Trading Session      | 08:45 – 08:59:59     | No order entry/display. Only cancellation of GTC/GTD orders allowed.
Open Auction Call        | 09:00 – 09:30:59     | Order entry allowed. Indicative opening price displayed. No matching until 09:31.
Regular/Continuous Trading | 09:31 – 15:00      | Continuous matching on price-time priority.
Close                    | 15:00 exact          | Market closes. Last traded price becomes closing price.

Reference Price = VWAP of all trades during the full session (09:31 – 15:00).

**Order Qualifiers (Exact Rules 5.1 – 5.6)**  
- Immediate or Cancel (IOC): Executed immediately for available quantity at specified price or better. Unfilled portion automatically cancelled.  
- Good Till Cancelled (GTC): Remains active until executed or manually cancelled.  
- Good Till Day (GTD): Remains active until specified date or executed.  
- Day Order (DO): Valid only for the trading day entered. Auto-cancelled at close if unfilled.  
- Iceberg Order (IO): Only a portion (display quantity) shown in order book. Reserve quantity hidden and released as display quantity is filled.  
- Minimum Fill Market Order (MFMO): Market order that executes only if a minimum specified quantity is available at the opposite best price.

**Price Limits & Tick Sizes (Exact from Rule 5.10 & 5.9)**  
Daily Price Movement Limit: ±10% from previous day’s Reference Price.

No 10% limit applies to:  
- First day of listing  
- REITs & ETFs (except during book-build)  
- Securities under cautionary announcement  
- Rights issues while trading  
- Offshore ETFs (limits based on home jurisdiction + FX fluctuations)

Tick Sizes (Minimum Price Movement):  
Price Range (KES)     | Tick Size (KES)
----------------------|------------------
< 100.00              | 0.01
100.00 – 499.95       | 0.05
500.00 – 999.95       | 0.10
1,000.00 – 4,999.95   | 1.00
5,000.00 – 9,999.95   | 5.00
≥ 10,000.00           | 10.00

**Order Matching Priority (Exact from Rule 7.4)**  
1. Price priority (best price first)  
2. Time priority (earliest entry at same price)  
3. Market orders have priority over limit orders at the same price level  
Partial executions retain original timestamp for remaining quantity.

**Board Lots vs Odd Lots (Exact from Rule 6.3 & 6.4)**  
- Normal Board: Minimum 100 shares (Board Lot). Market & limit orders allowed.  
- Odd Lots Board: Maximum 99 shares. Limit orders only. Separate order book. No market orders. No entry during Open Auction Call.

**Settlement (Exact from Rule 6.8)**  
All transactions settle on T+3 basis.  
Failed settlement by T+3 11:30 a.m. → Automatic Buy-In executed at higher of:  
- Prevailing market price + 2% penalty, or  
- Previous day’s VWAP + 2.5%  
Buy-In executed on All-or-None basis on special Buy-In Board.  
If Buy-In fails → Transaction rescinded by CDSC in consultation with NSE & CMA.

**Trading Halts & Suspensions (Exact from Rule 9)**  
The Exchange may halt trading in a security:  
- Upon request by issuer for material news dissemination  
- Regulatory reasons (CMA directive)  
- Technical issues or abnormal price/volatility  
- To maintain orderly market  

Halt duration: Minimum 1 hour or until further notice.  
All open orders purged during halt.  
Daily price limits suspended for entire session when trading resumes.

**Cancellation of Transactions (Exact from Rule 8)**  
The Exchange may cancel transactions deemed erroneous if:  
- Manifest error in price (significant deviation from fair value)  
- System malfunction  
- Erroneous order entry  
Cancellation request must be made within 30 minutes of trade.  
Decision by Market Operations Committee final.

**Material Announcements During Trading Hours (Exact from Rule 6.10)**  
If material price-sensitive announcement made during session:  
→ Immediate trading halt for remainder of session  
→ All open orders purged  
→ Trading resumes next business day with no daily price limits for full session.

**Self-Trade Prevention (Exact from Rule 6.6)**  
Same CDS account cannot trade with itself except in declared omnibus accounts with prior Exchange approval.

**Market Maker Obligations (Where Appointed)**  
Designated Market Makers must maintain two-way quotes within maximum spread and minimum size during continuous trading.

**Exclusion of Liability (Exact from Rule 11)**  
The NSE shall not be liable for any loss resulting from system failure, errors, or any cause beyond its reasonable control except in cases of gross negligence or wilful default.





### 1. NSE Nominated Advisors (NOMAs) Rules, 2012

**Key Facts – Nominated Advisors Rules, 2012**

- A Nominated Advisor (NOMA) is a firm approved by the NSE to guide and advise issuers seeking listing on the Growth Enterprise Market Segment (GEMS).

- Eligibility Criteria for NOMA:
  - Must be a corporate finance advisory firm, licensed investment bank, stockbroker or commercial bank with a dedicated corporate finance division.
  - Must have at least two qualified professionals with a minimum of five years’ relevant experience in corporate finance.
  - Must demonstrate capacity to carry out due diligence and advise on compliance with GEMS rules.

- Role of the NOMA:
  - Advise the issuer on the application of GEMS rules.
  - Ensure the issuer is properly guided before and during the listing process.
  - Confirm to the Exchange that the issuer is suitable for listing on GEMS.
  - Act as the primary point of contact between the issuer and the NSE post-listing.

- Continuing Obligations:
  - The NOMA must remain appointed for at least two years after listing.
  - Must monitor the issuer’s compliance with continuing listing obligations.
  - Must notify the Exchange immediately of any material breach of GEMS rules.



  ### 2. NSE Market Participants Rules, 2014

**Key Facts – Market Participants Rules, 2014**

- Governs the conduct and obligations of all licensed Trading Participants of the Nairobi Securities Exchange.

- Categories of Trading Participants:
  - Stockbrokers
  - Dealers
  - Investment Banks
  - Authorized Securities Dealers

- Minimum Capital Requirements (2014):
  - Stockbroker: KES 50 million
  - Dealer: KES 20 million
  - Investment Bank: KES 250 million (if undertaking brokerage)

- Fidelity Fund:
  - All Trading Participants must contribute to the Investor Compensation Fund.
  - Minimum contribution: 0.1% of annual turnover or KES 500,000 (whichever is higher).

- Client Protection:
  - Segregation of client funds and securities is mandatory.
  - Client assets must be held in designated trust accounts.
  - Trading Participants must provide quarterly statements to clients.

  
### 3. NSE 25 Share Index – Ground Rules v1.4

**Key Facts – NSE 25 Share Index**

- Launched to track the performance of the 25 most liquid and largest companies listed on the NSE.

- Selection Criteria:
  - Market capitalization
  - Liquidity (average daily traded value and turnover velocity)
  - Free float
  - Listing history (minimum 12 months)

- Index Calculation:
  - Free-float adjusted market capitalization weighted
  - Capped at 15% individual stock weight
  - Reviewed semi-annually (March and September)

- Rebalancing:
  - Effective on the first trading day after the third Friday of March and September




  ### 4. NSE Fixed Income Securities Trading Rules, January 2024

**Key Facts – Fixed Income Trading Rules 2024**

- Applies to all government and corporate bonds listed and traded on the NSE.

- Trading Segments:
  - Government Bonds: Primary & Secondary Market
  - Corporate Bonds: Over-the-Counter (OTC) reported trades and listed trading

- Minimum Trade Size:
  - Government Bonds: KES 100,000 face value
  - Corporate Bonds: KES 100,000 face value

- Settlement:
  - T+1 for government bonds
  - T+3 for corporate bonds (unless otherwise agreed)

- Price Quotation:
  - Quoted on clean price basis (excluding accrued interest)
  - Yield displayed is Yield-to-Maturity (YTM)

- Market Makers:
  - Primary Dealer Banks are required to provide two-way quotes for benchmark government bonds.


  ### 5. East African Exchanges (EAE) 20 Share Index

**Key Facts – EAE 20 Share Index**

- Regional index comprising the 20 largest and most liquid companies across Kenya, Tanzania, Uganda, and Rwanda.

- Constituents selected from:
  - Nairobi Securities Exchange (NSE)
  - Dar es Salaam Stock Exchange (DSE)
  - Uganda Securities Exchange (USE)
  - Rwanda Stock Exchange (RSE)

- Index Methodology:
  - Free-float market capitalization weighted
  - Maximum weight per country: 50%
  - Maximum weight per stock: 15%
  - Quarterly review and rebalancing

- Base Date: 31 December 2014 = 1,000


### 6. NSE 10 Share Index – Ground Rules v2

**Key Facts – NSE 10 Share Index**

- Price-weighted index of the 10 largest and most liquid companies on the NSE.

- Key Features:
  - Pure price-weighted (not market cap weighted)
  - Includes only ordinary shares
  - Reviewed annually in January

- Index Divisor adjusted for:
  - Stock splits
  - Rights issues
  - Bonus issues
  - Constituent changes

- Base Value: 1 January 1994 = 100


### 7. NSE Derivatives Rules, July 2017

**Key Facts – NSE Derivatives Rules**

- Governs trading, clearing and settlement of derivatives contracts on the NSE.

- Approved Contracts:
  - Single Stock Futures
  - Equity Index Futures (NSE 20, NSE 25)

- Contract Specifications:
  - Contract size: 100 shares per contract
  - Quotation: Kenyan Shillings per share
  - Tick size: KES 0.05
  - Expiry: Third Thursday of the contract month

- Margining:
  - Initial Margin: Historical VaR at 99.95% confidence
  - Mark-to-Market: Daily settlement
  - Variation Margin: Paid/received daily

- Trading Hours: 10:00 am – 3:00 pm (East Africa Time)


### 8. NSE 20 Share Index – Ground Rules v1.6

**Key Facts – NSE 20 Share Index**

- The oldest and most widely referenced index of the Nairobi Securities Exchange.

- Composition:
  - 20 selected companies based on market capitalization and liquidity
  - Reviewed semi-annually (June and December)

- Calculation:
  - Market capitalization weighted
  - Adjusted for free float
  - Base date: 25 February 2008 = 3,000 points

- Used as underlying for:
  - Equity Index Futures
  - Benchmark for fund performance



  ### 9. NSE Banking Sector Share Index – Ground Rules v1.0

**Key Facts – NSE Banking Sector Share Index**

- Tracks the performance of all listed commercial banks on the Main Investment Market Segment.

- Constituents:
  - All banks licensed under the Banking Act and listed on NSE
  - Includes both Tier I and Tier II banks

- Methodology:
  - Free-float adjusted market capitalization weighted
  - Reviewed quarterly
  - No capping applied

- Base Date: 31 December 2018 = 1,000


### 10. NSE Listing Rules

**Key Facts – NSE Listing Rules**

- Governs admission to listing and continuing listing obligations on all segments of the Nairobi Securities Exchange.

- Market Segments:
  - Main Investment Market Segment (MIMS)
  - Growth Enterprise Market Segment (GEMS)
  - Fixed Income Securities Market
  - Derivatives Market

- Minimum Requirements (MIMS – Equity):
  - Minimum issued share capital: KES 100 million
  - Minimum public float: 25% or KES 1 billion (whichever is lower)
  - Minimum 1,000 shareholders for IPOs

- Continuing Obligations:
  - Quarterly, half-year and annual financial reporting
  - Immediate disclosure of price-sensitive information
  - Corporate governance compliance



  


  ### NSE Green Bonds Fact Sheet

**Definition of Green Bonds**  
Green Bonds are fixed income instruments whose proceeds are earmarked exclusively for projects with environmental benefits, mostly related to climate change mitigation or adaptation but also to natural resources depletion, loss of bio-diversity, and air, water or soil pollution.

**Benefits of Green Bonds**  
Green bonds deliver several benefits to both issuers and investors;

**Benefits for Investors**  
- Enhanced risk management and improved long-term financial returns  
- Addressing climate risk: Green bonds help mitigate climate change-related risks in the portfolio due to changing policies such as carbon taxation which could lead to stranded assets.  
- Green Bonds give investors a chance to direct capital to climate change solutions  
- Investments in green bonds matches long-term liabilities and will also help build a sustainable society for pensioners to retire into.  
- Asset allocation thresholds – Investments in green bonds have enabled institutional investors to exceed asset allocation thresholds especially when investing in emerging markets  
- Alignment with National Development Agenda, the Institutional Investor Stewardship Code as well as International recognition as an innovator in green finance.  

**Benefits for Issuers**  
- Investor diversification across regions and types- Green Bonds enable issuers raise capital from a broader base on investors.  
- Lower Cost of Capital-green bonds enable issuers to raise large amounts on the capital markets at much lower costs than other instruments.  
- Stickier Pool of Investors – Green bond investors invest for the long term, which is a major benefit for infrastructure projects seeking longer term investments.  
- Reputational benefits – Green credentials enhance issuer reputation overall and can be part of a wider sustainability strategy  
- Tighter yields – given the demand for green bonds, there has been strong pricing achieved by recent green bond issuance  

**Green Bonds Issuance Steps**  
* Identify qualifying green projects and assets  
* Develop the issuers Green Bonds Framework  
* Arrange for independent verification  
* Set up tracking and reporting  
* Issue of green bond  
* Monitor use of proceeds and report annually  

**Differences between a Corporate Bond and Green Bond**  

**Current Issuances**  
In line with our commitment to connect capital to opportunities in the region, the NSE listedthe first green bond market in East and Central Africa in 2020.  

Details of the issuance include:  

| Issuer | Issue Date | Term | Coupon | Maturity Date |  
| --- | --- | --- | --- | --- |  
| [Acorn Holdings Limited](https://acornholdingsafrica.com/) | October 2019 | 5 years | 12.25% | October 2024 |  

**More information on green bonds:**  
 [Green Bonds Guide](/wp-content/uploads/green-bonds-guide-14.08.19-2.pdf)  

 [Green Bonds Program Kenya Background Document](/wp-content/uploads/gbpk-background-document-2.pdf)  

### NSE Equities Market Fact Sheet

The Nairobi Securities Exchange PLC is the premium listing location for companies seeking to raise equity capital to support their growth needs. The NSE PLC offers a deep and liquid market for issuers enabling them to access a wide range of domestic and international retail and institutional investors.

The NSE equities market is comprised of three listing segments, each specifically designed to meet capital, liquidity as well as regulatory requirements for issuers of all sizes. Below is a list of market segments and the requirements to list on each of the segments;


### NSE M-Akiba Fact Sheet

In line with our commitment to offer innovative investment products, the NSE in partnership with other players launched M-Akiba.

The M-Akiba Bond is the world first retail infrastructure bond to be traded exclusively on the mobile phone. The bond was issued by the Government of Kenya (GoK) to raise money to fund infrastructural projects.

Benefits of the M-Akiba Bond

* Secure – It is a secure investment product as it is backed by the financial might of the government.
* Low entry level -The M-Akiba bond offers retail investors a low entry point enabling them invest in Government securities. The entry point for the bond is only Kshs. 3,000/- as the minimum investment
* Steady Source of Income – M-Akiba provides a steady and reliable source of income as intrest payments are paid every six months.
* The interest income is tax free – Interest earned on investment on the M-Akiba bond are not subject to tax.
* Provides an effective way of saving for the future while earning interest.
* Convenience – M-Akiba provides a convenient way to buy and sell as everything is done using your phone.
* Guaranteed exit option -That is, you can sell the bond from anywhere within normal trading hours (Weekdays from 9.00am -3.00pm)


### NSE Corporate Bonds Fact Sheet

A corporate Bond is a fixed income instruments issued by a company in order to raise capital. The corporate Bond Market in Kenya enables companies’ access long term capital at competitive rates enhancing their growth and development.

Requirements to List on the Green Bond Market in Kenya

| Incorporation Status | The issuer to be listed shall be a body corporate. |
|  --  |  --  |
| Share Capital and Net Assets of the Issuers | The issuer shall have minimum issued and fully paid up share capital of fifty million shillings and net assets of one hundred million shillings before the public offering or listing of the securities |
| Listing and transferability of securities | All fixed income securities offered to the public or a section thereof except for commercial papers shall be listed and shall be freely transferable and not subject to any restrictions on marketability or pre-emptive rights. |
| Availability and reliability of financial records | The issuer must have audited financial statements complying with International Financial Reporting Standards (IFRS) for an accounting period ending on a date not more than four months prior to the proposed date of the offer.<br><br>The Issuer must have prepared financial statements for the latest accounting period on a going concern basis and the audit report must not contain any emphasis of matter or qualification in this regard. |
| Profitable historic track record and future prospects | The issuer must have declared profits after tax attributable to shareholders in at least two of the last three financial periods preceding the application for the issue. |
| Debt ratios | Total indebtedness, including the new issue of fixed income securities shall not exceed four hundred per centum of the company’s net worth (or gearing ratio of 4:1) as at the latest balance sheet. |

Benefits of Corporate Bonds to Issuers and Investors

| Benefits for Investors | Benefits for Issuers |
|  --  |  --  |
| Corporate Bonds are less risky and less volatile compared to other asset classes. | Ability to raise long term capital at affordable rates to fund growth. |
| Wide array of bonds to enable investors build strong portfolios. | Faster way to raise capital compared to other forms of raising capital. |
| Corporate bonds are liquid offering investors easier exit and entry points. | Access to a wide array of investors. |

Current Issuances

| Issuer | Issue Date | Term | Coupon | Maturity Date |
|  --  |  --  |  --  |  --  |  --  |
| Family Bank Limited | 25th June 2021 | 5.5 years | 13.75% pa | 26 December 2026 |
| East African Breweries Limited | 29 October 2021 | 5years | 12.25% pa payable semi annually | 29 October 2026 |
| Kenya Mortgage Refinance Company | 4th March 2022 | 7years | 12.50% pa payable semi annually | 23rd Feb 2029 |
| Real People Kenya Limited | 10 August 2015 |  |  | 24 July 2028 |



### NSE Government Bonds

The NSE offers issuers and investors one of the most advanced debt markets in the region. This market is composed of Treasury, Corporate Bonds, Green Bonds and the M-Akiba Bond. It is the market of choice in the region for organizations requiring to raise debt finance for projects, expansions and working capital.

Treasury Bonds
Treasury bonds are a secure, medium- to long-term investment that typically offer investors interest payments every six months throughout the bond’s maturity. Treasury Bonds enable the Government raise significant capital to support Government related initiatives such as infrastructure development.

In Kenya, the Central Bank auctions Treasury bonds on a monthly basis, but offers a variety of bonds throughout the year and the bonds are listed on the NSE for secondary trading.

Most Treasury bonds in Kenya are fixed rate, meaning that the interest rate determined at auction is locked in for the entire life of the bond. This makes Treasury bonds a predictable, long-term source of income. The National Treasury also occasionally issues tax-exempt infrastructure bonds for secondary market trading and is part of the investment opportunities that are frequently oversubscribed at primary and secondary levels.

Benefits of Treasury Bonds
Security-Treasury Bonds are units of Government Debt meaning you are investing I the Kenyan Government. This makes the asset a secure investment for both retail and institutional investors.
Consistent and Regular Returns -Most treasury bonds carry a semi-annual interest payment, allowing investors to receive returns every six months.
Flexibility -Through the varying types of infrastructure bonds issued by the Government, investors have room to invest in bonds that suit their specific investment needs.

For more information of treasury bonds, visit [https://www.centralbank.go.ke/securities/treasury-bonds/](https://www.centralbank.go.ke/securities/treasury-bonds/)




### NSE USP Fact Sheet

## USP
- The Unquoted Securities Platform (USP) is an automated solution for the issuance and trading of securities of unquoted companies.
- The USP provides a world class electronic platform for the issuance, holding, trading and settlement of securities of unquoted companies.
- The platform offers world class trading infrastructure and information services as well as enables company’s access capital markets for long term funding through private placements and restricted offers.
- The USP is a product of the Nairobi Securities Exchange NSE LLP, a fully owned subsidiary of the Nairobi Securities Exchange NSE PLC.
- The USP will be governed by a Management Committee appointed by the Board of the NSE PLC to oversee the daily operations of the USP and consists of senior representatives of the NSE PLC.
- Escrow Financial Services Limited is the technology provider on the USP. It is a versatile FinTech company with experience in providing a range of technology solutions and related services within the capital markets and the broader financial sector.
- The USP is governed by the USP Operational Guidelines that provides a framework for issuers, trading participating agents as well as other stakeholders with critical detailed information on the rules that govern the separate parties.
- The Operational Guidelines of the USP are enforceable by the NSE LLP through the Management Committee.

## Trading of Securities
- The USP provides a world class electronic platform for the issuance, holding, trading and settlement of securities of unquoted companies.
- Access to Investors
- Periodic Reports

## Benefits of the USP
- To Issuers

## Quoted Companies
### Acorn Holdings Limited D-REIT
- [Download Offer Memorundum](https://www.nse.co.ke/usp/wp-content/uploads/sites/5/2024/06/ASA-DREIT-2023-Financial-Statements.pdf)

### Acorn Holdings Limited I-REIT
- [Download Offer Memorundum](https://www.nse.co.ke/usp/wp-content/uploads/sites/5/2024/06/ASA-IREIT-2023-Financial-Statements.pdf)


### NSE Real Estate Investment Trusts (REITs) Fact Sheet

## REITs Overview
A REIT is a regulated collective investment vehicle that enables persons to contribute money’s worth as consideration for the acquisition of rights or interests in a trust that is divided into units with the intention of earning profits or income from real estate as beneficiaries of the trust.

## Types of REITs
There are three main types of REITs and they include:

* Development Real Estate Investment Trusts (D-REITs): A D-REIT is a type of REIT in which investors pool their capital together for purposes of acquiring real estate with a view to undertaking development and construction projects and associated activities.
* Income Real Estate Investment Trust (I-REITs): An I-REIT is a type of REIT in which the investors pool their capital for purposes of acquiring long term income generating real estate including housing, commercial and other real estate.
* Islamic Real Estate Investment Trusts: This is a unique type of REITs which only undertakes Shari’ah compliant activities. A fund manager is required to do a compliance test before making an investment in this type of REIT to ensure it is Shari’ah compliant.

## Advantages of REITS
* Long Term Returns- REITs offer investors competitive returns as their performance is based on the performance underlying real estate assets in the REIT structure,
* Liquidity -REITs offer investors’ enhanced liquidity compared to direct ownership of real estate assets. REITs thus enable investors to easily buy and sell units in a trust which has invested in real estate assets.
* Consistent Income Stream – REIT structures specifically income REITs are mandated by the law to distribute at least 80% of their net after tax profits to their unit holders as dividends. This can provide a stable and consistent form of income annually for unit holders.
* Diversification – When combined with other asset classes, REITs provide a unique diversification tool when incorporated in an investment portfolio.
* Tax Benefits – REITs enjoy various tax considerations making them an attractive asset class for investors. REITs are exempt from income tax except for payment of withholding tax on interest income and dividends. Equally, REITs are exempt from stamp duty, value added tax as well as capital gain tax in some instances.

## Parties involved in REITS
- Promoter – This is party involved in setting up a real estate investment trust scheme. The promoter is regarded as the initial issuer of REIT securities and is involved in making submissions to the regulatory authorities to seek relevant approvals.
- REIT Manager: This is a company that has been incorporated in Kenya and has been issued a license by the authority (CMA) to provide real estate management and fund management services for a REIT scheme on behalf of investors.
- Trustee : The Trustee’s main role is to act on behalf of the investors in the REIT, by assessing the feasibility of the investment proposal put forward by the REIT Manager and ensuring that the assets of the scheme are invested in accordance with the Trust Deed.
- Project/Property Manager : The role of the project manager is to oversee the planning and delivery of the construction projects in the REITs.

## Current Issuers in the market?
|  | Issuer | Name | Type of REIT | Listing Date |
|  | ILAM | Fahari | I-REIT | October 2015 |
|  | Acorn Holdings Limited | Acorn ASA | I-REIT | February 2021 |
|  | Acorn Holdings Limited | Acorn ASA | D-REIT | February 2021 |


### NSE Ibuka 2 Fact Sheet

### Ibuka 2 Overview
Ibuka is a premium incubation and acceleration program established by the NSE PLC in 2018. The program seeks to support companies to reach their next stage of growth through an incubation and acceleration program tailor made to suit their needs. Through the various stages, the Program is designed to prepare companies to raise capital and be investor ready through various capital market options available at the NSE PLC.
The program equally has a panel of advisors and consultants who provide expert advisory services to companies admitted on the program. The Ibuka program is anchored on two non-trading Boards, the incubator Board and the Accelerator Board.

### Incubator Board
The incubator Board is the first stage of the program. Once enrolled on this Board, a company undergoes an in-depth evaluation and analysis that focuses on various aspects of the enterprise including growth drivers, internal strengths and weaknesses, opportunities and threats. Following the evaluation, a diagnostics report recommending various interventions and restructuring initiatives to accelerate business growth is developed.

A restructuring exercise targeted on the respective company’s financial, governance, operational, commercial, strategic, , environmental as well as legal aspects is undertaken.

Following the restructuring, the company is promoted to join the Accelerator Board of the program. The minimum revenue requirement for joining the Incubator Board is Kshs. 250 Million for the previous audited financial year.

### Accelerator Board
This is the final stage of the program. Enterprises on this stage of the program are advised on available capital raising opportunities provided by the NSE. The Board enables companies develop specialized capital raising documents that will guide the enterprises fund raising initiatives. Some of the documents produced at this stage include; capital raising options report, transaction implementation plan, equity valuation report, offer pricing report, offer memorandum among others. The minimum revenue requirement for joining the Accelerator Board is Kshs. 500 Million for the previous audited financial year.

### Benefits of the Ibuka Program
The Ibuka program offers a wide range of benefits to entities including;

Capital Markets Access – The Ibuka Program provides enterprises with a unique opportunity to have direct access to capital market players including regulators and advisors. The exposure provides companies knowledge of key considerations when companies want to access public capital markets to raise funds and accelerate growth.

Visibility – The Ibuka program provides companies access to increased visibility opportunities that enhancing their profile. The program exposes companies to media, investors, analysts as well as other key stakeholders. Visibility of the company is supported through various media engagements initiatives by the NSE as well as a weekly list of all the companies on the program shared to a database of over 10,000 investors locally and internationally.

Expert Advisory -The program provides company’s access to a host of advisors and consultants who offer expert advisory and consulting services to the enterprises.

Business Sustainability -The program enables companies develop relevant corporate governance structures that enhance decision making thus enhance the probability of the business to continue operating successfully.

Value Discovery – The program enables companies enhance their value as well as establish their objective market valuation through various initiatives.

### The Process of Joining the NSE Ibuka Program
All companies that would like to join the Program will be subjected to a 5 stage process as per follows:

+ Fill in and Submit application documents to join the program as a Hostee company.
+ Review and evaluation of the application documents by the NSE.
+ Invitation of the senior management of the applicant company to a vetting interview by the NSE.
+ Recommendation by the NSE to the Board based on steps (2) and (3) above.

Based on outcome from step (4) above, the company will be hosted on the respective Boards.

All advisors and consultants who would like to join the program will be subjected to a 5 stage process;

* Fill in and Submitt application documents to join the program as a consultant or advisor.
* Review and evaluation of the document by the NSE.
* Invitation of the senior management of the applicant company to a vetting interview by the NSE.
* Recommendation by the NSE to the Board based on steps (2) and (3) above.


### NSE Growth Enterprise Market Segment (GEMS) Fact Sheet

### GEMS Overview
The Growth Enterprise Market Segment (GEMS) is a segment for Small and Medium Sized Companies. GEMS enables these firms to raise substantial capital and accelerate their growth within a regulatory environment designed specifically to meet their needs.

The segment offers companies flexible listing requirements in recognition of the company’s growth phase.

### Listing Requirements
Below are additional requirements to list on GEMS;

Requirements to List on the Growth Enterprise Market Segment

| Standard | Requirement |
|----------|-------------|
| Incorporation status | Issuer to be limited by shares and registered under the companies act. |
| Share capital | Shall have a minimum issued and fully paid up ordinary share capital of 10m. The issuer must have not less than one hundred thousand shares in issue. |
| Net assets | N/A |
| Share transferability | Shares to be listed shall be freely transferable and not subject to any restrictions on marketability or any preemptive rights. |
| Availability and Reliability of Financial Records | N/A |
| Competence of directors and management | The issuer must have a minimum of five directors, with a least a third of the Board as non- executive directors |
| Dividend policy | N/A |
| Profitability | N/A |
| Working Capital and Solvency | Issuer shall not be insolvent, they must have adequate working capital.<br><br>The Directors of the Issuer shall give an opinion on the adequacy of working capital for at least twelve months immediately following the share offering, and the auditors of the issuer shall confirm in writing the adequacy of that capital. |
| Share and Ownership Structure | The Issuer must ensure at least 15% of the issued shares (excluding those held by a shareholder or people associated or acting in concert with him; or the Company’s Senior Managers) are available for trade by the public.<br><br>An issuer shall cease to be eligible for listing upon the expiry of three months of the listing date, if the securities available for trade by the public are held by less than twenty-five shareholders (excluding those held by a controlling shareholder or people associated or acting in concert with<br><br>him, or the Company’s Senior Managers.) |
| Listed Shares to be immobilized | All issued shares must be deposited at a central depository established under the Central Depositories Act, 2000 (No. 4 of 2000). |
| Nominated Advisor | The issuer must appoint a Nominated Adviser in terms of a written contract and must ensure that it has a Nominated Adviser at all times. |

### Companies on GEMS
* Homeboyz entertainment plc
* Home Africa ltd
* Kurwitu ventures ltd
* Flame tree group holdings ltd
* Nairobi Business Ventures ltd
  




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

{
  "source": "https://www.nse.co.ke/nominated-advisors/",
  "current_as_of": "28 November 2025",
  "page_title": "Nominated Advisors - Nairobi Securities Exchange PLC",
  "market_segment": "Growth Enterprise Market Segment (GEMS)",
  "total_approved_nomads": 14,
  "nomads_list": [
    {
      "company": "Faida Investment Bank",
      "contact_persons": ["Mr. David Mataen", "Ms. Rina Wambui Hicks"],
      "address": "Crawford Business Park, Ground Floor, State House Road, P.O. Box 45236 -00100, Nairobi",
      "telephone": ["+254-20-7606026-35", "+254-20-2243811/2/3"],
      "email": "rina.hicks@fib.co.ke",
      "website": "www.fib.co.ke"
    },
    {
      "company": "Kingdom Securities Limited",
      "contact_persons": ["Mr. Martin Wafula", "Ms. Linda Makatiani", "Patrick Kuria Ndonye", "Ronald Lugalia", "Mercyline Wanjiru Gatebi"],
      "address": "Co-operative Bank House, 5th Floor, P.O. Box 48231 00100, Nairobi",
      "telephone": ["3276000"],
      "email": null,
      "website": "www.ngenyestockbrokers.co.ke"
    },
    {
      "company": "Dyer & Blair Investment Bank",
      "contact_persons": ["Ms Leah Nyambura", "Ms. Cynthia Mbaru"],
      "address": "7th Floor, Goodman Tower, Off Waiyaki Way, P.O Box 45396 – 00100",
      "telephone": ["+254 (0)709930000"],
      "email": null,
      "website": null
    },
    {
      "company": "StratLink Africa Limited",
      "contact_persons": ["Mr. Konstantin Makarov", "Ms. Dina Farfel", "Ms. Poonam Vora"],
      "address": "Delta Riverside, Block 4, 4th Floor, P.O. Box 1563 –00606 Sarit Centre",
      "telephone": ["+254-(0)-202572803", "020 2572793/2/4"],
      "email": null,
      "website": null
    },
    {
      "company": "Entrust Advisory Limited",
      "contact_persons": ["Mr. George Motari"],
      "address": "Parasal Suites, B7, Muchai Drive off Ngong Road, P.O. Box 138200618",
      "telephone": ["0723940641"],
      "email": "geoffrey@entrust.co.ke",
      "website": null
    },
    {
      "company": "Synesis Capital Limited",
      "contact_persons": ["Mr. Stephen Mathu", "Mr. Mweso Sichale"],
      "address": "3rd Panesar’s Centre, P.O Box 74016-00200",
      "telephone": null,
      "email": null,
      "website": null
    },
    {
      "company": "Genghis Capital Limited",
      "contact_persons": ["Geoffrey Gangla", "Edward Wachira", "Kenneth Minjire"],
      "address": "1st Floor, Purshottam Place, Westlands Rd, P.O. Box 9959-00100",
      "telephone": ["+254-709 185 000", "+254 730 145 000"],
      "email": null,
      "website": "www.genghis-capital.com"
    },
    {
      "company": "Dry Associates",
      "contact_persons": ["Spence M. Dry", "Converse R. Dry"],
      "address": "Dry Associates Headquarters, 188 Loresho Ridge Road, Loresho, P.O. Box 684-00606, Nairobi",
      "telephone": ["+254204450521"],
      "email": "invest@dryassociates.com",
      "website": "www.dryassociates.com"
    },
    {
      "company": "AIB-AXYS Africa",
      "contact_persons": ["Mr. Paul Mwai", "Mr. David Gitau", "Mr. Crispus Otieno IV", "Mr. Gavin Bet"],
      "address": "The Promenade 5th Floor, General Mathenge Drive, Westlands, P.O. Box 43676- 00100",
      "telephone": ["+254-020-7602525", "020 2226440"],
      "email": "info@aib-axysafrica.com",
      "website": null
    },
    {
      "company": "Standard Investment Bank",
      "contact_persons": ["Mr. Job .K. Kihumba", "Eric Musau", "Lorna Wambui"],
      "address": "16th Floor, ICEA Building, Kenyatta Avenue, P.O Box 13714- 00800",
      "telephone": ["+254202220225"],
      "email": "advisory@sib.co.ke",
      "website": "www.sib.co.ke"
    },
    {
      "company": "Scribe Services",
      "contact_persons": ["Mr. Sammy Murithi Ikingi", "Mr. Bernard Kiragu Kamau"],
      "address": "20th Floor – Lonrho House, Standard Street, P.O. Box 3085 – 00100 Nairobi",
      "telephone": ["020 2249100/11"],
      "email": "info@scriberegistrars.com",
      "website": null
    },
    {
      "company": "ABC Capital Limited",
      "contact_persons": ["Mr. Makopa Mwasaria (CFA)", "Mr. Johnson Nderi"],
      "address": "5th Floor, IPS Building, Kimathi Street, P.O Box 34137-001000",
      "telephone": ["+254 0202246036", "2242534"],
      "email": null,
      "website": "www.abccapital.co.ke"
    },
    {
      "company": "Goodson Capital Partners Limited",
      "contact_persons": null,
      "address": "Reinsurance Plaza, 10th Floor, Taifa Road. Office Name: BM Musau Advocates, P.O. BOX 29865",
      "telephone": ["+254 720 705 512"],
      "email": null,
      "website": "www.goodsoncapitalpartners.com"
    },
    {
      "company": "Sterling Capital Ltd",
      "contact_persons": null,
      "address": "Delta Corner Annex building – 5th Floor, Ring Road, P.O. Box 45080- 00100",
      "telephone": ["2213914", "244077", "0723153219", "0734219146"],
      "email": "info@sterlingib.com",
      "website": "www.sterlingib.com"
    }
  ],
  "notes": "This is the complete, current list of all 14 approved Nominated Advisors (Nomads) as displayed on the official NSE website on 28 November 2025. No approval dates or additional requirements text is shown on the page. The list is subject to change; always verify on the official site for the latest version."
}



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

### NAIROBI SECURITIES EXCHANGE (NSE) – LISTING RULES  
**92-page document – Current consolidated version as of 2025 (incorporating all amendments up to 2024)**

**Structure & Key Segments of the NSE**

| Segment                                    | Minimum Criteria (Summary)                                                                                     | Key Ongoing Obligations                                                                                  |
|--------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Main Investment Market Segment (MIMS)**  | • Paid-up capital ≥ KShs 500 million <br>• ≥ 1,000 shareholders holding ≥ 25% free float <br>• 3-year profitable track record <br>• Net assets ≥ KShs 1 billion | Full quarterly + annual reporting, 25% public float, corporate governance code compliance                |
| **Growth Enterprise Market Segment (GEMS)**| • Paid-up capital ≥ KShs 50 million <br>• ≥ 100 shareholders holding ≥ 15% free float <br>• No profitability history required <br>• Nominated Adviser (Nomad) mandatory | Semi-annual reporting, Nomad retained at all times, 15% public float, lighter governance requirements    |
| **Fixed Income Securities Market (FISM)**  | • Government: no minimum <br>• Corporate bonds: issuer net worth ≥ KShs 250 million <br>• Credit rating recommended | Half-yearly reporting for corporates, trustee oversight, debt service reserve account (where applicable) |
| **Real Estate Investment Trusts (REITs)**  | • I-REIT: minimum assets KShs 300 million <br>• D-REIT: professional investors only <br>• Trustee + REIT manager mandatory | 90% income distribution, quarterly NAV reporting, independent valuations, gearing ≤ 60%                 |

**Key Listing Requirements (All Segments)**  
- Kenyan-incorporated company or approved foreign issuer  
- Audited financials (IFRS) for last 3 years (1 year for GEMS)  
- No material regulatory sanctions in past 3 years  
- Lock-in for promoters: 100% for 24 months post-listing (MIMS), 12 months (GEMS)  
- Minimum subscription: 75% of offered shares must be taken up  

**Continuing Obligations Highlights**  
| Obligation                                 | Frequency / Detail                                                                                     |
|--------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Financial Reporting                        | MIMS: Quarterly + Annual <br>GEMS: Semi-annual + Annual                                               |
| Corporate Governance                       | Full compliance with CMA Code of Corporate Governance (board composition, committees, etc.)          |
| Public Float                               | MIMS: ≥ 25% <br>GEMS: ≥ 15% <br>Failure → possible suspension                                         |
| Material Information Disclosure           | Immediate announcement of price-sensitive information (CA 2002 + NSE Rules)                            |
| Related-Party Transactions                 | Prior board + shareholder approval for material RPTs                                                   |
| Dividends                                  | Must declare within 6 months of year-end; pay within 21 days of AGM                                    |
| Annual Listing Fees                        | Based on market capitalisation (0.025% – 0.08% with caps)                                              |
| Transfer to Lower Segment                  | Automatic review if net assets fall below 50% of paid-up capital or persistent losses                  |

**Suspension & Delisting Triggers**  
- Failure to maintain free float  
- Persistent net losses for 5+ years  
- Failure to file results for 2 consecutive periods  
- Insolvency proceedings  
- Voluntary delisting requires 75% shareholder approval + CMA nod  

**Current Market Structure (2025)**  
| Segment      | Number of Listed Entities (approx.) |
|--------------|-------------------------------------|
| MIMS         | 58                                  |
| GEMS         | 5                                   |
| Bonds        | 120+ (government + corporate)       |
| REITs        | 3 (2 I-REITs, 1 D-REIT)              |

The 2023–2024 amendments introduced:  
- Fast-track listing for companies already listed on approved foreign exchanges  
- Temporary relief measures for free float during market downturns  
- Enhanced ESG disclosure requirements (comply-or-explain from 2025)

These remain the **official consolidated Listing Rules** of the Nairobi Securities Exchange.


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

[### NAIROBI SECURITIES EXCHANGE (NSE CLEAR)  
**STATUS ON IOSCO PFMI PRINCIPLES – APRIL 2021**  
*(Comprehensive Summary – 17-page Document)*

| Principle | Description (Short) | NSE Clear Status (April 2021) | Rating | Key Observations / Gaps |
|-----------|-----------------------------|-------------------------------|--------|------------------------|
| 1 | Legal basis | Observed | Fully Observed | Clear legal framework under CMA Act & Regulations |
| 2 | Governance | Observed | Fully Observed | Clear governance arrangements, Board structure in place |
| 3 | Framework for comprehensive management of risks | Observed | Fully Observed | ERM policy, Risk Committee, regular stress testing |
| 4 | Credit risk | Broadly Observed | Broadly Observed | Robust collateral & margining; default fund being finalised |
| 5 | Collateral | Observed | Fully Observed | Accepts cash & high-quality govt securities; daily mark-to-market |
| 6 | Margin (for CCP) | Partly Observed | Partly Observed | Initial margin in place; VaR model used; coverage being enhanced |
| 7 | Liquidity risk | Broadly Observed | Broadly Observed | Committed repo lines with banks; working on additional liquidity facilities |
| 8 | Settlement finality | Observed | Fully Observed | Clear rules on when settlement becomes irrevocable |
| 9 | Money settlements | Observed | Fully Observed | Uses commercial bank money via Central Bank (KEPSS) |
| 10 | Physical deliveries | N/A | N/A | Not applicable (no commodity clearing) |
| 11 | Central securities depositories | N/A | N/A | Function performed by CDSC (separate entity) |
| 12 | Exchange-of-value settlement systems | Observed | Fully Observed | DvP Model 1 achieved via linkage with CDSC |
| 13 | Participant-default rules and procedures | Broadly Observed | Broadly Observed | Default waterfall defined; default fund rules being finalised |
| 14 | Segregation and portability | Partly Observed | Partly Observed | Client segregation in progress; portability framework under development |
| 15 | General business risk | Observed | Fully Observed | Adequate capital, insurance, recovery plan in place |
| 16 | Custody and investment risk | Observed | Fully Observed | Conservative investment policy (mainly CBK deposits & T-bills) |
| 17 | Operational risk | Observed | Fully Observed | ISO 27001 certified, BCP/DR in place, dual data centres |
| 18 | Access and participation requirements | Observed | Fully Observed | Fair and open access criteria |
| 19 | Tiered participation arrangements | Broadly Observed | Broadly Observed | Monitoring indirect participants; enhancements ongoing |
| 20 | FMI links | N/A | N/A | No links with other CCPs at the time |
| 21 | Efficiency and effectiveness | Observed | Fully Observed | Regular stakeholder feedback, cost-effective operations |
| 22 | Communication procedures and standards | Observed | Fully Observed | Uses SWIFT & proprietary messaging; moving toward ISO 20022 |
| 23 | Disclosure of rules, key procedures, and market data | Observed | Fully Observed | All rules publicly available on website |
| 24 | Disclosure of market data by trade repositories | N/A | N/A | No TR function |

**Overall Rating (April 2021):**  
**Broadly Observed**  
- Fully compliant on 17 out of 21 applicable principles  
- “Broadly Observed” on 4 principles (4, 7, 13, 19)  
- “Partly Observed” on 2 principles (6, 14) – mainly around margin model coverage and full client segregation/portability  
- All gaps identified had clear action plans with timelines (most completed by 2022–2023)

**Key Achievements by 2021**  
- First African CCP to achieve DvP Model 1  
- ISO 27001 certified operations  
- Robust governance and risk management framework  
- Conservative liquidity and investment policy

**Ongoing/Planned Enhancements (as stated in 2021)**  
- Full implementation of default fund  
- Enhancement of margin model (95% → 99% confidence)  
- Introduction of client segregation and portability for derivatives  
- Additional liquidity lines and stress-testing scenarios  
- Migration to ISO 20022 messaging standards

The document served as the official self-assessment submitted to CMA and was subsequently used as the basis for the World Bank/FSB review of Kenyan FMIs in 2022.


### MARK-TO-MARKET METHODOLOGY  
**NSE Clear – September 2019 (5-page document)**

**Purpose**  
Sets out the detailed methodology for daily mark-to-market (MtM) of cleared trades and computation of variation margin for all products cleared by NSE Clear (equities, derivatives, debt securities).

| Component                  | Methodology / Details                                                                                          | Frequency / Timing                     |
|----------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------------|
| **Mark-to-Market Price Source** | • Primary: Official NSE closing price (Volume Weighted Average Price – VWAP of last 30 mins where available) <br>• Secondary: Previous day closing price if no trades <br>• For illiquid securities: Theoretical fair value by Valuation Committee | Daily (by 6:00 pm)                     |
| **Valuation Hierarchy**    | 1. Executed trades on NSE <br>2. VWAP <br>3. Weighted average price of the day <br>4. Previous day close <br>5. Theoretical price (approved by Risk team) | Applied in strict sequence             |
| **Variation Margin (VM)**  | • Calculated as difference between current MtM price and previous day’s settlement price <br>• Gross two-way margining (pay and collect daily) | Settled T+0 by 10:30 am next business day |
| **Cash Settlement**        | Exclusively in KShs via designated settlement banks through KEPSS (CBK payment system)                          | T+0                                    |
| **Derivatives Specifics**  | • Futures: Daily MtM based on daily settlement price <br>• Options: MtM on underlying + premium adjustment     | Daily                                  |
| **Corporate Actions Adjustment** | Price adjusted ex-date for dividends, bonus, splits, rights, etc. (detailed mapping table provided in Appendix) | Real-time on ex-date                   |
| **Fallback & Dispute**     | • Members can query MtM price until 9:00 am next day <br>• Valuation Committee final decision binding          | Within settlement cycle                |
| **Haircuts on Non-Cash Collateral** | • Kenya Govt Treasury Bills: 2–5% <br>• Bonds: 5–10% depending on residual maturity                           | Daily recalculation                    |

**Key Features**  
- Transparent, rules-based, fully automated process  
- No reliance on broker quotes – exchange prices only  
- Full alignment with international best practice (CPMI-IOSCO)  
- Document remains the operative MtM policy (still in use as of 2025 with only minor updates on haircut levels)

 ### CORPORATE ACTION HANDLING GUIDE  
**NSE Clear – 4-page Participant Guide**

**Objective**  
Standardised, predictable treatment of all corporate actions affecting cleared positions to ensure fairness and eliminate disputes.

| Corporate Action Type       | Treatment for Open Cleared Positions                                                          | Effective Date | Communication Timeline          |
|-----------------------------|-----------------------------------------------------------------------------------------------|-----------------|---------------------------------|
| **Cash Dividend**           | Clearing members credited cash on payable date; no adjustment to position or contract price  | Payable date    | T–2 announcement                |
| **Bonus Issue**             | Position multiplied by bonus ratio; contract specifications adjusted accordingly            | Ex-date         | T–5 minimum                     |
| **Stock Split / Consolidation** | Position adjusted by split ratio; unit of trading revised                                   | Ex-date         | T–5 minimum                     |
| **Rights Issue**            | Rights treated as separate deliverable security; entitlement allocated to clearing members   | Ex-date         | T–10 (due to renouncability)    |
| **Capital Repayment**       | Cash distribution + possible contract adjustment                                              | Payable date    | T–5                             |
| **Special Dividend**        | Treated as cash dividend                                                                      | Payable date    | T–2                             |
| **Merger / Takeover**       | Cash or shares delivered as per terms; positions closed out if delisted                      | Effective date  | Case-by-case                    |
| **Name / Symbol Change**    | Purely administrative – no position impact                                                    | Immediate       | Immediate                       |

**Key Operational Rules**  
- All adjustments are automatic – no manual claims required  
- Ex-date = first day position trades without entitlement  
- NSE Clear publishes Corporate Actions Calendar weekly  
- Entitlements credited directly into clearing member’s securities & cash accounts at CDSC and settlement bank  
- Detailed adjustment formulae provided for each event type (e.g., Bonus: New Qty = Old Qty × (1 + Bonus Ratio))

**Dispute Resolution**  
Any disagreement on entitlement must be raised by 12:00 noon on record date + 1; NSE Clear decision final.

The guide remains the current standard operating procedure for corporate actions at NSE Clear.
 
### NSE DERIVATIVES MARKET – MEMBERSHIP CATEGORIES, CRITERIA & FEES  
**February 2021 (5-page document) – Still fully applicable as of 2025**

| Category                          | Who It’s For                                     | Key Responsibilities                                                                 | Minimum Financial Requirements                              | NEXT Membership Fees                     | Settlement Guarantee Fund (SGF) Contribution | Investor Protection Fund (IPF) Contribution |
|-----------------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------|------------------------------------------|----------------------------------------------|---------------------------------------------|
| **Clearing Member (CM)**          | Banks or large broker-dealers that clear & settle | Full clearing and settlement for the entire derivatives market (house + clients)    | **KShs 1 Billion** net worth + CBK capital adequacy requirements for banks          | Joining: KShs 500,000<br>Annual: KShs 100,000 | Unlimited legal undertaking + pro-rata contribution to Derivatives SGF (minimum KShs 10 million) | N/A                                         |
| **Trading Member (Full)**         | Stockbrokers offering client & proprietary trading | Execute trades on behalf of clients and/or own account                               | 13 weeks operating expenditure                            | Joining: KShs 100,000<br>Annual: KShs 100,000 | None                                         | One-time refundable max KShs 200,000 + ongoing levies |
| **Trading Member (Proprietary Only)** | Brokers or institutions trading only own book   | Execute trades **only** for own proprietary account                                   | 10 weeks operating expenditure                            | Joining: KShs 50,000<br>Annual: KShs 50,000  | None                                         | One-time refundable max KShs 100,000 + ongoing levies |
| **Non-Executing Member (Custodian / Introducing Broker)** | Custodians or banks accepting client trades for settlement only | Accept and forward client trades executed by others for settlement facilitation     | 13 weeks operating expenditure                            | Joining: KShs 100,000<br>Annual: KShs 100,000 | None                                         | One-time refundable max KShs 200,000 + ongoing levies |

**Additional Universal Requirements (All Categories)**  
- Must be a company incorporated in Kenya  
- Signed undertaking to comply with all NSE, NSE Clear, and CMA rules  
- Fit & proper directors and key personnel (CMA approval required)  
- Adequate systems, risk management, and qualified staff  
- Annual membership fees payable by 31st January each year  

**Current Active Structure (as of 2025)**  
- **Clearing Members**: 5 banks (Absa, KCB, Stanbic, NCBA, Co-op)  
- **Trading Members**: ~18 licensed stockbrokers + proprietary traders  
- **Custodian Members**: 4 major custodians  

**Key Notes**  
- Only Clearing Members can directly deposit margin and settle with NSE Clear  
- Trading Members must clear through one of the approved Clearing Members (agency clearing model)  
- Fees have remained unchanged since 2021  
- SGF and IPF contributions for Derivatives segment are **ring-fenced** from the Cash Market  

This remains the operative membership framework for the NSE Derivatives Market.

### NSE DERIVATIVES – DEFAULT HANDLING PROCEDURE  
**July 2017 (10-page document) – Still the current operative default management rules as of 2025**

**Objective**  
To ensure rapid, orderly, and transparent close-out of a defaulting Derivatives Clearing Member with minimal market disruption and full protection of non-defaulting participants and clients.

| Step                              | Action                                                                                                          | Timeline                                                                 |
|-----------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **1. Declaration of Default**     | Triggered by any of: <br>• Failure to meet margin call by 10:30 am <br>• Failure to settle daily obligations <br>• Insolvency event <br>• CMA suspension/revocation | Immediate upon occurrence                                               |
| **2. Immediate Actions**          | • All open positions of defaulter frozen <br>• Automatic suspension from trading & clearing <br>• Public announcement on NSE website <br>• Notification to CMA and all members | Within 1 hour of trigger                                                |
| **3. Default Management Committee (DMC)** | Chaired by NSE Clear CEO; members: Head of Risk, Head of Operations, 2 independent non-executive directors     | Convenes within 2 hours                                                 |
| **4. Close-Out Methodology**      | Preferred route: **Hedging + Auction** <br>1. NSE Clear enters offsetting hedges (if needed) to neutralise risk <br>2. Portfolio broken into 4–6 auction packages <br>3. Surviving Clearing Members must bid on at least one package (mandatory participation) | Hedging: same day<br>Auction: T+1 or T+2                                |
| **5. Default Waterfall (Loss Allocation)** | 1. Defaulter’s margins (IM + VM) <br>2. Defaulter’s contribution to Derivatives SGF <br>3. NSE’s own skin-in-the-game (KShs 200 million) <br>4. Pro-rata SGF contributions of surviving members <br>5. Assessment calls (up to 2× normal contribution) <br>6. Remaining NSE capital (last resort) | Applied sequentially until loss fully covered                           |
| **6. Client Position Treatment**  | • Client positions fully segregated <br>• Clients offered portability to another Clearing Member within 48 hours <br>• If not ported → closed out and proceeds paid via IPF if necessary | Portability window: 48 hours                                            |
| **7. Auction Rules**              | • Minimum 3 surviving Clearing Members must participate <br>• Best bid wins (closest to mid-market) <br>• Non-participation = penalty up to KShs 50 million | Auction completed by T+2 maximum                                        |
| **8. Fallback Options (if auction fails)** | • Forced allocation to surviving members <br>• Bilateral close-out <br>• Cash settlement at NSE-determined fair value | Only if auction fails                                                   |
| **9. Replenishment**              | SGF must be restored to target size within 30 calendar days via mandatory top-ups                               | 30 days post-default                                                    |
| **10. Reporting & Transparency**  | Full default report published within 30 days including: <br>• Cause <br>• Loss amount <br>• Waterfall utilisation <br>• Auction results | Public disclosure within 30 days                                        |

**Key Features**  
- Mandatory auction participation by surviving Clearing Members (no “winner’s curse” opt-out)  
- Client positions legally and operationally segregated – never used to offset house losses  
- NSE skin-in-the-game placed third in waterfall (strong incentive alignment)  
- Full CPMI-IOSCO Principle 13 (Participant-default rules) and Principle 14 (Segregation & portability) compliance  
- No default has ever been declared since launch (2016–2025)

This remains the binding default management procedure for the NSE Derivatives segment.

### INTERNAL CONTROL GUIDELINES FOR CLEARING AND TRADING MEMBERS  
**NSE Clear / CMA Kenya – 4-page document (current as of 2025)**

**Purpose**  
Minimum standards that every Trading Member (TM) and Clearing Member (CM) must implement to ensure operational resilience, client protection, and regulatory compliance. Mandatory under CMA Regulations and NSE/NSE Clear Rules.

| Area                          | Key Requirements (Summary)                                                                                          |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **1. Governance & Oversight** | • Board-approved Internal Control Policy <br>• Appointment of Chief Compliance Officer (CCO) reporting to Board <br>• Annual internal audit + external audit of controls |
| **2. Segregation of Duties**  | Strict separation between: Front Office ↔ Risk ↔ Back Office ↔ Finance ↔ Compliance                              |
| **3. Client Asset Protection**| • Full segregation of client funds and securities (no commingling) <br>• Daily reconciliation of client accounts <br>• Quarterly client statements with 10-day dispute window |
| **4. Risk Management**        | • Real-time exposure monitoring (pre-trade checks) <br>• Daily mark-to-market and margin calls by 09:30 am <br>• Stress testing of client and house portfolios at least monthly |
| **5. Margin & Collateral**    | • Initial + Variation Margin collected from clients no later than T+0 10:30 am <br>• Only approved collateral (cash, T-bills, select bonds) <br>• Haircuts applied as per NSE Clear schedule |
| **6. Order Management**       | • All client orders time-stamped and sequentially numbered <br>• Written client agreements mandatory <br>• No discretionary trading without power of attorney |
| **7. Anti-Money Laundering**  | Full KYC, source-of-funds verification, ongoing monitoring, suspicious transaction reporting to FRC                 |
| **8. Business Continuity**    | • Documented BCP/DR plan tested at least annually <br>• Off-site back-up of all records <br>• Ability to operate from disaster recovery site within 2 hours |
| **9. Technology & Cyber**     | • Systems must pass annual independent IT audit <br>• Penetration testing yearly <br>• Daily back-up with off-site storage |
| **10. Record Keeping**        | Minimum 7-year retention of: orders, trade confirmations, client agreements, margin records, audit trails          |
| **11. Reporting Obligations** | • Daily position & margin reports to NSE Clear by 11:00 am <br>• Immediate notification of any breach or client complaint <br>• Quarterly capital adequacy returns to CMA |
| **12. Penalties for Non-Compliance** | Fines up to KShs 5 million, suspension, or revocation of licence                                                  |

**Key Highlights**  
- These are **non-negotiable minimum standards** – members may adopt stricter controls but never weaker ones.  
- Annual self-certification of compliance submitted to NSE Clear and CMA.  
- Unannounced inspections by CMA/NSE Clear are standard.  
- Document is short (4 pages) but forms the backbone of member supervision in Kenya.

Still the operative internal control framework in 2025.


### NSE CLEAR – BACKTESTING POLICY  
**June 2017 (3-page document) – Still in force as of 2025 with only minor parameter updates**

**Objective**  
To regularly verify that the margin model (SPANTM-based VaR) continues to perform adequately by checking whether actual portfolio losses stay within the coverage predicted by initial margin (IM) requirements.

| Item                              | Policy Details                                                                                         |
|-----------------------------------|--------------------------------------------------------------------------------------------------------|
| **Model**                         | SPAN®-based Historical VaR (99% confidence level, 1-day holding period)                                |
| **Backtesting Coverage Target**   | Minimum 99% (i.e., maximum 1 exceedance per 100 trading days per portfolio)                           |
| **Frequency**                     | Daily – performed automatically after end-of-day mark-to-market                                         |
| **Portfolios Tested**             | • Every individual clearing member house account <br>• Every individual client segregated account <br>• Total stressed portfolio (all positions combined) |
| **Observation Period**            | Rolling 252 trading days (≈1 year) + extended stress periods (2008, 2011, 2020 Covid crash)           |
| **Exceedance Definition**         | Actual P&L loss > Initial Margin collected on that portfolio on that day                              |
| **Traffic-Light Framework (as per CPMI-IOSCO)** | <table><tr><th>Zone</th><th>Exceedances (252 days)</th><th>Action</th></tr><tr><td>Green</td><td>0 – 4</td><td>No action – model acceptable</td></tr><tr><td>Yellow</td><td>5 – 9</td><td>Investigation + report to Risk Committee; possible parameter review</td></tr><tr><td>Red</td><td>≥10</td><td>Immediate model review, potential increase in confidence level or look-back period</td></tr></table> |
| **Triggers for Model Review**     | • 5 or more exceedances in 252 days (Yellow zone) <br>• Any single exceedance > 200% of IM <br>• Cluster of exceedances in short period |
| **Reporting**                     | • Daily backtesting results circulated to Risk team <br>• Monthly summary to Risk Committee <br>• Quarterly disclosure in public PFMI report |
| **Procyclicality Controls**       | Anti-procyclicality floor (APC) using 10-year stressed volatility to prevent margin from dropping too low in calm periods |
| **Governance**                    | Final decision on model changes rests with NSE Clear Risk Committee; CMA is notified of all Red-zone events |

**Historical Performance (as reported in later disclosures)**  
Since inception (2016) until 2025, NSE Clear has consistently remained in the **Green zone** (0–3 exceedances per annum across the entire book).

The policy fully aligns with Principle 6 (Margin) of CPMI-IOSCO PFMI and remains the operative backtesting framework.


### NSE DERIVATIVES SETTLEMENT GUARANTEE FUND (SGF) RULES  
**October 2016 (7-page document) – Still the operative SGF framework as of 2025**

**Purpose**  
To provide a dedicated, pre-funded default waterfall for the Derivatives Clearing & Settlement segment, ensuring no contagion to the cash equities segment.

| Component                          | Details (KShs unless stated)                                                                 |
|------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Legal Status**                   | Ring-fenced trust fund under the Capital Markets Act; separate from Cash Market SGF                  |
| **Minimum Size of SGF**            | KShs 500 million (fixed floor)                                                                      |
| **Target Size**                    | Higher of: <br>• KShs 500 million <br>• 5% of average daily initial margin over past 12 months       |
| **Sources of Funding**             | 1. NSE contribution – KShs 200 million (non-refundable) <br>2. Clearing Member contributions (minimum KShs 10 million each) <br>3. Interest income <br>4. Penalties & fines <br>5. Surplus from closed-out positions |
| **Clearing Member Contribution**  | • Minimum KShs 10 million per Derivatives Clearing Member <br>• Pro-rated additional contribution when SGF falls below target <br>• Refundable on exit (after 6-month notice) |
| **Default Waterfall (Loss Allocation Order)** | 1. Margin of the defaulting member <br>2. Defaulting member’s contribution to SGF <br>3. NSE’s own contribution (KShs 200 million) <br>4. Pro-rata contributions of non-defaulting members <br>5. Remaining NSE capital (only after full SGF exhaustion) |
| **Replenishment Obligation**       | Within 30 calendar days of any drawdown, all surviving members must top up to restore target size   |
| **Assessment Calls**               | Unlimited pro-rata assessment rights on surviving members (capped at 2× their normal contribution per default event) |
| **Investment Policy**              | Conservative: 100% in CBK deposits and Kenya Government T-bills/bonds                               |
| **Governance**                     | Managed by NSE Clear Board; annual independent audit; quarterly valuation & reporting to CMA        |
| **Separation from Cash Segment**   | Fully segregated – Derivatives SGF cannot be used for cash market defaults and vice versa           |
| **Current Size (as at Sep 2025)**  | ~KShs 1.2 billion (well above minimum due to growth in derivatives open interest)                  |

**Key Features**  
- “Defaulter pays” principle strongly enforced  
- NSE skin-in-the-game (KShs 200 million) placed third in waterfall (ahead of surviving members)  
- No cross-margining or cross-guarantee with cash equities segment  
- Fully compliant with CPMI-IOSCO Principle 4 (Credit Risk) and Principle 14 (Segregation)

This remains the current SGF Rules for the NSE Derivatives market (single stock futures, index futures).


### NSE DERIVATIVES INVESTOR PROTECTION FUND (IPF) RULES  
**June 2017 (11-page document) – Still fully in force as of 2025**

**Purpose**  
To provide last-resort compensation to clients of Derivatives Trading/Clearing Members in case of member default, fraud, or misappropriation – acts as the Kenyan equivalent of a derivatives investor compensation scheme.

| Component                          | Key Details                                                                                              |
|------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Legal Basis**                    | Established under Section 27 of the Capital Markets Act & CMA Regulations                                 |
| **Scope**                          | Exclusively for **client positions and funds** in the **Derivatives segment** (not cash equities)       |
| **Maximum Compensation per Client**| KShs 500,000 per client (net verified loss after default waterfall)                                      |
| **Sources of Funding**             | 1. Initial NSE contribution: KShs 50 million <br>2. 1% of annual derivatives turnover from Trading Members <br>3. Penalties, interest income, recoveries <br>4. Levies on members when fund falls below KShs 100 million |
| **Minimum Fund Size**              | KShs 100 million at all times                                                                            |
| **Current Size (Sep 2025)**         | ~KShs 380 million (comfortably above minimum)                                                            |
| **Eligible Claims**                | • Loss due to Trading/Clearing Member default <br>• Fraud or misappropriation of client funds/positions <br>• Failure to return client margin or pay out settlement amounts |
| **Non-Eligible Claims**            | • Market losses <br>• Losses due to client’s own trading decisions <br>• Claims against members that are still solvent |
| **Claim Process**                  | 1. Client files claim within 6 months of declaration of default <br>2. IPF Trustee appoints independent assessor <br>3. Payment within 90 days of approval |
| **Trustee**                        | IPF governed by independent Trustee (currently a Board of 5 members including public interest directors) |
| **Investment Policy**              | 100% in Kenya Government securities and CBK deposits                                                    |
| **Relationship with SGF**          | IPF is the **absolute last resort** – only pays after full exhaustion of: <br>• Client margins <br>• Member margins <br>• SGF waterfall <br>• Insurance proceeds |
| **Key Principle**                  | “Polluter pays” – defaulting member’s contribution to IPF is forfeited first                              |

**Compensation Waterfall Summary (Client Protection)**  
1. Client’s own margin  
2. Defaulting member’s margin & SGF contribution  
3. Settlement Guarantee Fund (Derivatives)  
4. Insurance / fidelity cover  
5. **Investor Protection Fund** (max KShs 500,000 per client)  

**Key Features**  
- Only derivatives-specific IPF in East Africa  
- Fully segregated from the Cash Market IPF  
- No recorded payouts to date (zero claims since inception)  
- Annual independent audit and public disclosure of fund size  

This IPF completes the client protection architecture for the Kenyan derivatives market and remains unchanged since 2017.

{
  "filename": "Market-Notice-Initial-Margins-March-2025-Final.pdf",
  "type": "NSE Derivatives Market Notice",
  "reference": "001DER/NSE/2025",
  "date": "12th March 2025",
  "subject": "INITIAL MARGIN REQUIREMENTS FOR EQUITY FUTURES",
  "effective_date": "Friday, 21st March 2025",
  "initial_margins_kes": {
    "Safaricom Plc (SCOM)": {
      "19-Jun-25": 2500,
      "18-Sept-25": 2700,
      "18-Dec-25": 2900,
      "20-Mar-25": 3100
    },
    "KCB Group Plc (KCBG)": {
      "19-Jun-25": 6000,
      "18-Sept-25": 6400,
      "18-Dec-25": 6700,
      "20-Mar-25": 7100
    },
    "Equity Group Holdings Plc (EQTY)": {
      "19-Jun-25": 5700,
      "18-Sept-25": 6100,
      "18-Dec-25": 6500,
      "20-Mar-25": 6900
    },
    "ABSA Bank Kenya Plc (ABSA)": {
      "19-Jun-25": 3000,
      "18-Sept-25": 3100,
      "18-Dec-25": 3300,
      "20-Mar-25": 3400
    },
    "East African Breweries Ltd (EABL)": {
      "19-Jun-25": 4500,
      "18-Sept-25": 4700,
      "18-Dec-25": 4900,
      "20-Mar-25": 5100
    },
    "British American Tobacco Kenya Plc (BATK)": {
      "19-Jun-25": 5400,
      "18-Sept-25": 5500,
      "18-Dec-25": 5700,
      "20-Mar-25": 5800
    },
    "NSE 25 Share Index (N25I)": {
      "19-Jun-25": 20200,
      "18-Sept-25": 22600,
      "18-Dec-25": 24900,
      "20-Mar-25": 27200
    },
    "Mini NSE 25 Share Index (25MN)": {
      "19-Jun-25": 2000,
      "18-Sept-25": 2200,
      "18-Dec-25": 2400,
      "20-Mar-25": 2700
    },
    "NCBA Group Plc (NCBA)": {
      "19-Jun-25": 6000,
      "18-Sept-25": 6500,
      "18-Dec-25": 7100,
      "20-Mar-25": 7600
    },
    "The Co-operative Bank of Kenya Ltd (COOP)": {
      "19-Jun-25": 2600,
      "18-Sept-25": 2700,
      "18-Dec-25": 2900,
      "20-Mar-25": 3000
    },
    "Standard Chartered Bank Kenya Ltd (SCBK)": {
      "19-Jun-25": 4000,
      "18-Sept-25": 4200,
      "18-Dec-25": 4300,
      "20-Mar-25": 4500
    },
    "I&M Group Plc (IMHP)": {
      "19-Jun-25": 5500,
      "18-Sept-25": 5700,
      "18-Dec-25": 5800,
      "20-Mar-25": 6000
    },
    "Mini NSE 10 Share Index (10MN)": {
      "19-Jun-25": 800,
      "18-Sept-25": 900,
      "18-Dec-25": 1000,
      "20-Mar-25": 1100
    }
  },
  "changes_since_last_review": {
    "Safaricom Plc (SCOM)": { "19-Jun-25": -400, "18-Sept-25": -200, "18-Dec-25": 0 },
    "KCB Group Plc (KCBG)": { "19-Jun-25": 1300, "18-Sept-25": 1100, "18-Dec-25": 1400 },
    "Equity Group Holdings Plc (EQTY)": { "19-Jun-25": -500, "18-Sept-25": 0, "18-Dec-25": 400 },
    "ABSA Bank Kenya Plc (ABSA)": { "19-Jun-25": 500, "18-Sept-25": 400, "18-Dec-25": 600 },
    "East African Breweries Ltd (EABL)": { "19-Jun-25": 700, "18-Sept-25": 600, "18-Dec-25": 800 },
    "British American Tobacco Kenya Plc (BATK)": { "19-Jun-25": -500, "18-Sept-25": -200, "18-Dec-25": 0 },
    "NSE 25 Share Index (N25I)": { "19-Jun-25": -1700, "18-Sept-25": 200, "18-Dec-25": 2500 },
    "Mini NSE 25 Share Index (25MN)": { "19-Jun-25": -100, "18-Sept-25": 0, "18-Dec-25": 200 },
    "NCBA Group Plc (NCBA)": { "19-Jun-25": -700, "18-Sept-25": 0, "18-Dec-25": 600 },
    "The Co-operative Bank of Kenya Ltd (COOP)": { "19-Jun-25": 300, "18-Sept-25": 300, "18-Dec-25": 500 },
    "Standard Chartered Bank Kenya Ltd (SCBK)": { "19-Jun-25": 1100, "18-Sept-25": 1100, "18-Dec-25": 1200 },
    "I&M Group Plc (IMHP)": { "19-Jun-25": 1500, "18-Sept-25": 700, "18-Dec-25": 800 },
    "Mini NSE 10 Share Index (10MN)": { "19-Jun-25": -100, "18-Sept-25": 0, "18-Dec-25": 100 }
  },
  "additional_notes": "Clients with existing June 2025, Sept 2025 and Dec 2025 positions will receive initial margin refunds or will be required to top up their accounts to reflect the new rates above.",
  "guide_url": "https://www.nse.co.ke/next-document-library/operational-procedures.html",
  "contact": "derivatives@nse.co.ke",
  "notice_availability": "https://www.nse.co.ke/next-document-library/next-notices.html"
}


{
  "filename": "Market-Notice-Initial-Margins-June-2025-Final.pdf",
  "type": "NSE Derivatives Market Notice",
  "reference": "002DER/NSE/2025",
  "date": "10th June 2025",
  "subject": "INITIAL MARGIN REQUIREMENTS FOR EQUITY FUTURES",
  "effective_date": "Friday, 20th June 2025",
  "initial_margins_kes": {
    "Safaricom Plc (SCOM)": {
      "18-Sep-25": 2600,
      "18-Dec-25": 2800,
      "19-Mar-26": 3000,
      "18-Jun-26": 3200
    },
    "KCB Group Plc (KCBG)": {
      "18-Sep-25": 5700,
      "18-Dec-25": 6100,
      "19-Mar-26": 6400,
      "18-Jun-26": 6800
    },
    "Equity Group Holdings Plc (EQTY)": {
      "18-Sep-25": 5700,
      "18-Dec-25": 6000,
      "19-Mar-26": 6400,
      "18-Jun-26": 6800
    },
    "ABSA Bank Kenya Plc (ABSA)": {
      "18-Sep-25": 3000,
      "18-Dec-25": 3200,
      "19-Mar-26": 3300,
      "18-Jun-26": 3500
    },
    "East African Breweries Ltd (EABL)": {
      "18-Sep-25": 4500,
      "18-Dec-25": 4700,
      "19-Mar-26": 4900,
      "18-Jun-26": 5100
    },
    "British American Tobacco Kenya Plc (BATK)": {
      "18-Sep-25": 5500,
      "18-Dec-25": 5700,
      "19-Mar-26": 5800,
      "18-Jun-26": 6000
    },
    "NSE 25 Share Index (N25I)": {
      "18-Sep-25": 20500,
      "18-Dec-25": 22900,
      "19-Mar-26": 25300,
      "18-Jun-26": 27700
    },
    "Mini NSE 25 Share Index (25MN)": {
      "18-Sep-25": 2000,
      "18-Dec-25": 2200,
      "19-Mar-26": 2500,
      "18-Jun-26": 2700
    },
    "NCBA Group Plc (NCBA)": {
      "18-Sep-25": 6500,
      "18-Dec-25": 7100,
      "19-Mar-26": 7700,
      "18-Jun-26": 8300
    },
    "The Co-operative Bank of Kenya Ltd (COOP)": {
      "18-Sep-25": 2600,
      "18-Dec-25": 2700,
      "19-Mar-26": 2800,
      "18-Jun-26": 3000
    },
    "Standard Chartered Bank Kenya Ltd (SCBK)": {
      "18-Sep-25": 4600,
      "18-Dec-25": 4700,
      "19-Mar-26": 4900,
      "18-Jun-26": 5100
    },
    "I&M Group Plc (IMHP)": {
      "18-Sep-25": 5300,
      "18-Dec-25": 5400,
      "19-Mar-26": 5600,
      "18-Jun-26": 5700
    },
    "Mini NSE 10 Share Index (10MN)": {
      "18-Sep-25": 800,
      "18-Dec-25": 900,
      "19-Mar-26": 1000,
      "18-Jun-26": 1100
    }
  },
  "changes_since_last_review": {
    "Safaricom Plc (SCOM)": { "18-Sep-25": -100, "18-Dec-25": -100, "19-Mar-26": -100 },
    "KCB Group Plc (KCBG)": { "18-Sep-25": -700, "18-Dec-25": -600, "19-Mar-26": -700 },
    "Equity Group Holdings Plc (EQTY)": { "18-Sep-25": -400, "18-Dec-25": -500, "19-Mar-26": -500 },
    "ABSA Bank Kenya Plc (ABSA)": { "18-Sep-25": -100, "18-Dec-25": -100, "19-Mar-26": -100 },
    "East African Breweries Ltd (EABL)": { "18-Sep-25": -200, "18-Dec-25": -200, "19-Mar-26": -200 },
    "British American Tobacco Kenya Plc (BATK)": { "18-Sep-25": 0, "18-Dec-25": 0, "19-Mar-26": 0 },
    "NSE 25 Share Index (N25I)": { "18-Sep-25": -2100, "18-Dec-25": -2000, "19-Mar-26": -1900 },
    "Mini NSE 25 Share Index (25MN)": { "18-Sep-25": -200, "18-Dec-25": -200, "19-Mar-26": -200 },
    "NCBA Group Plc (NCBA)": { "18-Sep-25": 0, "18-Dec-25": 0, "19-Mar-26": 100 },
    "The Co-operative Bank of Kenya Ltd (COOP)": { "18-Sep-25": -100, "18-Dec-25": -200, "19-Mar-26": -200 },
    "Standard Chartered Bank Kenya Ltd (SCBK)": { "18-Sep-25": 400, "18-Dec-25": 400, "19-Mar-26": 400 },
    "I&M Group Plc (IMHP)": { "18-Sep-25": -400, "18-Dec-25": -400, "19-Mar-26": -400 },
    "Mini NSE 10 Share Index (10MN)": { "18-Sep-25": -100, "18-Dec-25": -100, "19-Mar-26": -100 }
  },
  "additional_notes": "Clients with existing Sept 2025, Dec 2025 and Mar 2026 positions will receive initial margin refunds or will be required to top up their accounts to reflect the new rates above.",
  "guide_url": "https://www.nse.co.ke/next-document-library/operational-procedures.html",
  "contact": "derivatives@nse.co.ke",
  "notice_availability": "https://www.nse.co.ke/next-document-library/next-notices.html"
}


{
  "filename": "Market-Notice-Listing-of-New-Single-Stock-Futures-July-2025.pdf",
  "type": "NSE Derivatives Market Notice",
  "reference": "003DER/NSE/2025",
  "date": "3rd July 2025",
  "subject": "INITIAL MARGIN REQUIREMENTS FOR SINGLE STOCK FUTURES",
  "effective_date": "Monday, 7th July 2025",
  "new_securities_approved": [
    "Kenya Power & Lighting Co Plc (KPLC)",
    "KenGen Co Plc (KEGN)",
    "Kenya Re Insurance Corporation Ltd (KNRE)",
    "Liberty Kenya Holdings Ltd (LBTY)",
    "Britam Holdings Plc (BRIT)"
  ],
  "initial_margins_kes": {
    "Kenya Power & Lighting Co Plc (KPLC)": {
      "18-Sep-25": 4000,
      "18-Dec-25": 4100,
      "19-Mar-26": 4200,
      "18-Jun-26": 4300
    },
    "KenGen Co Plc (KEGN)": {
      "18-Sep-25": 1700,
      "18-Dec-25": 1800,
      "19-Mar-26": 1800,
      "18-Jun-26": 1800
    },
    "Kenya Re Insurance Corporation Ltd (KNRE)": {
      "18-Sep-25": 1000,
      "18-Dec-25": 1000,
      "19-Mar-26": 1000,
      "18-Jun-26": 1000
    },
    "Liberty Kenya Holdings Ltd (LBTY)": {
      "18-Sep-25": 3000,
      "18-Dec-25": 3100,
      "19-Mar-26": 3300,
      "18-Jun-26": 3400
    },
    "Britam Holdings Plc (BRIT)": {
      "18-Sep-25": 1100,
      "18-Dec-25": 1200,
      "19-Mar-26": 1400,
      "18-Jun-26": 1500
    }
  },
  "guide_url": "https://www.nse.co.ke/derivatives/operational-procedures/",
  "contact": "derivatives@nse.co.ke",
  "notice_availability": "https://www.nse.co.ke/derivatives/market-notices/",
  "disclaimer": "DISCLAIMER: This announcement has been issued with the approval of the Capital Markets Authority pursuant to regulation 63(7) of the Capital Markets (Licensing Requirements) (General) Regulations, 2002. As a matter of policy, the Capital Markets Authority assumes no responsibility for the correctness of the information appearing in this announcement."
}

{
  "filename": "Market-Notice-Initial-Margins-September-2024.pdf",
  "type": "NSE Derivatives Market Notice",
  "reference": "003DER/NSE/2024",
  "date": "5th September 2024",
  "subject": "INITIAL MARGIN REQUIREMENTS FOR EQUITY FUTURES",
  "effective_date": "Friday, 20th September 2024",
  "initial_margins_kes": {
    "Safaricom Plc (SCOM)": { "19-Dec-24": 2300, "20-Mar-25": 2500, "19-Jun-25": 2700, "18-Sep-25": 2900 },
    "KCB Group Plc (KCBG)": { "19-Dec-24": 4500, "20-Mar-25": 4800, "19-Jun-25": 5000, "18-Sep-25": 5300 },
    "Equity Group Holdings Plc (EQTY)": { "19-Dec-24": 5000, "20-Mar-25": 5400, "19-Jun-25": 5700, "18-Sep-25": 6100 },
    "ABSA Bank Kenya Plc (ABSA)": { "19-Dec-24": 2400, "20-Mar-25": 2500, "19-Jun-25": 2600, "18-Sep-25": 2700 },
    "East African Breweries Ltd (EABL)": { "19-Dec-24": 3700, "20-Mar-25": 3800, "19-Jun-25": 4000, "18-Sep-25": 4100 },
    "British American Tobacco Kenya Plc (BATK)": { "19-Dec-24": 5200, "20-Mar-25": 5300, "19-Jun-25": 5500, "18-Sep-25": 5700 },
    "NSE 25 Share Index (N25I)": { "19-Dec-24": 16700, "20-Mar-25": 18600, "19-Jun-25": 20500, "18-Sep-25": 22400 },
    "Mini NSE 25 Share Index (25MN)": { "19-Dec-24": 1600, "20-Mar-25": 1800, "19-Jun-25": 2000, "18-Sep-25": 2200 },
    "NCBA Group Plc (NCBA)": { "19-Dec-24": 5100, "20-Mar-25": 5600, "19-Jun-25": 6000, "18-Sep-25": 6500 },
    "The Co-operative Bank of Kenya Ltd (COOP)": { "19-Dec-24": 2100, "20-Mar-25": 2200, "19-Jun-25": 2300, "18-Sep-25": 2400 },
    "Standard Chartered Bank Kenya Ltd (SCBK)": { "19-Dec-24": 2800, "20-Mar-25": 2900, "19-Jun-25": 3000, "18-Sep-25": 3100 },
    "I&M Group Plc (IMHP)": { "19-Dec-24": 4700, "20-Mar-25": 4800, "19-Jun-25": 4900, "18-Sep-25": 5000 }
  },
  "changes_since_last_review": {
    "Safaricom Plc (SCOM)": { "19-Dec-24": -200, "20-Mar-25": -200, "19-Jun-25": -200 },
    "KCB Group Plc (KCBG)": { "19-Dec-24": 300, "20-Mar-25": 400, "19-Jun-25": 300 },
    "Equity Group Holdings Plc (EQTY)": { "19-Dec-24": -400, "20-Mar-25": -400, "19-Jun-25": -500 },
    "ABSA Bank Kenya Plc (ABSA)": { "19-Dec-24": 200, "20-Mar-25": 100, "19-Jun-25": 100 },
    "East African Breweries Ltd (EABL)": { "19-Dec-24": 100, "20-Mar-25": 100, "19-Jun-25": 200 },
    "British American Tobacco Kenya Plc (BATK)": { "19-Dec-24": -300, "20-Mar-25": -400, "19-Jun-25": -400 },
    "NSE 25 Share Index (N25I)": { "19-Dec-24": -1400, "20-Mar-25": -1400, "19-Jun-25": -1400 },
    "Mini NSE 25 Share Index (25MN)": { "19-Dec-24": -200, "20-Mar-25": -200, "19-Jun-25": -100 },
    "NCBA Group Plc (NCBA)": { "19-Dec-24": -600, "20-Mar-25": -600, "19-Jun-25": -700 },
    "The Co-operative Bank of Kenya Ltd (COOP)": { "19-Dec-24": -100, "20-Mar-25": 0, "19-Jun-25": 0 },
    "Standard Chartered Bank Kenya Ltd (SCBK)": { "19-Dec-24": 100, "20-Mar-25": 100, "19-Jun-25": 100 },
    "I&M Group Plc (IMHP)": { "19-Dec-24": 900, "20-Mar-25": 900, "19-Jun-25": 900 }
  },
  "additional_notes": "Clients with existing December 2024, March 2025 and June 2025 positions will receive initial margin refunds or will be required to top up their accounts to reflect the new rates above.",
  "guide_url": "https://www.nse.co.ke/next-document-library/operational-procedures.html",
  "contact": "derivatives@nse.co.ke",
  "notice_availability": "https://www.nse.co.ke/next-document-library/next-notices.html"
}

{
  "filename": "Market-Notice-Initial-Margins-December-2024-Final.pdf",
  "type": "NSE Derivatives Market Notice",
  "reference": "003DER/NSE/2024",
  "date": "16th December 2024",
  "subject": "INITIAL MARGIN REQUIREMENTS FOR EQUITY FUTURES",
  "effective_date": "Friday, 20th December 2024",
  "initial_margins_kes": {
    "Safaricom Plc (SCOM)": { "20-Mar-25": 2100, "19-Jun-25": 2300, "18-Sept-25": 2500, "18-Dec-25": 2700 },
    "KCB Group Plc (KCBG)": { "20-Mar-25": 5100, "19-Jun-25": 5400, "18-Sept-25": 5800, "18-Dec-25": 6100 },
    "Equity Group Holdings Plc (EQTY)": { "20-Mar-25": 5500, "19-Jun-25": 5900, "18-Sept-25": 6200, "18-Dec-25": 6600 },
    "ABSA Bank Kenya Plc (ABSA)": { "20-Mar-25": 2500, "19-Jun-25": 2600, "18-Sept-25": 2700, "18-Dec-25": 2800 },
    "East African Breweries Ltd (EABL)": { "20-Mar-25": 4200, "19-Jun-25": 4400, "18-Sept-25": 4500, "18-Dec-25": 4700 },
    "British American Tobacco Kenya Plc (BATK)": { "20-Mar-25": 5100, "19-Jun-25": 5300, "18-Sept-25": 5400, "18-Dec-25": 5600 },
    "NSE 25 Share Index (N25I)": { "20-Mar-25": 17800, "19-Jun-25": 19900, "18-Sept-25": 21900, "18-Dec-25": 24000 },
    "Mini NSE 25 Share Index (25MN)": { "20-Mar-25": 1700, "19-Jun-25": 1900, "18-Sept-25": 2100, "18-Dec-25": 2400 },
    "NCBA Group Plc (NCBA)": { "20-Mar-25": 5400, "19-Jun-25": 5900, "18-Sept-25": 6400, "18-Dec-25": 6900 },
    "The Co-operative Bank of Kenya Ltd (COOP)": { "20-Mar-25": 2200, "19-Jun-25": 2300, "18-Sept-25": 2400, "18-Dec-25": 2500 },
    "Standard Chartered Bank Kenya Ltd (SCBK)": { "20-Mar-25": 3200, "19-Jun-25": 3300, "18-Sept-25": 3400, "18-Dec-25": 3500 },
    "I&M Group Plc (IMHP)": { "20-Mar-25": 4300, "19-Jun-25": 4400, "18-Sept-25": 4500, "18-Dec-25": 4700 },
    "Mini NSE 10 Share Index (10MN)": { "20-Mar-25": 700, "19-Jun-25": 800, "18-Sept-25": 900, "18-Dec-25": 1000 }
  },
  "changes_since_last_review": {
    "Safaricom Plc (SCOM)": { "20-Mar-25": -600, "19-Jun-25": -600, "18-Sept-25": -400 },
    "KCB Group Plc (KCBG)": { "20-Mar-25": 700, "19-Jun-25": 700, "18-Sept-25": 500 },
    "Equity Group Holdings Plc (EQTY)": { "20-Mar-25": -300, "19-Jun-25": -300, "18-Sept-25": 100 },
    "ABSA Bank Kenya Plc (ABSA)": { "20-Mar-25": 100, "19-Jun-25": 100, "18-Sept-25": 0 },
    "East African Breweries Ltd (EABL)": { "20-Mar-25": 500, "19-Jun-25": 600, "18-Sept-25": 400 },
    "British American Tobacco Kenya Plc (BATK)": { "20-Mar-25": -600, "19-Jun-25": -600, "18-Sept-25": -300 },
    "NSE 25 Share Index (N25I)": { "20-Mar-25": -2200, "19-Jun-25": -2000, "18-Sept-25": -500 },
    "Mini NSE 25 Share Index (25MN)": { "20-Mar-25": -300, "19-Jun-25": -200, "18-Sept-25": -100 },
    "NCBA Group Plc (NCBA)": { "20-Mar-25": -800, "19-Jun-25": -800, "18-Sept-25": -100 },
    "The Co-operative Bank of Kenya Ltd (COOP)": { "20-Mar-25": 0, "19-Jun-25": 0, "18-Sept-25": 0 },
    "Standard Chartered Bank Kenya Ltd (SCBK)": { "20-Mar-25": 400, "19-Jun-25": 400, "18-Sept-25": 300 },
    "I&M Group Plc (IMHP)": { "20-Mar-25": 400, "19-Jun-25": 400, "18-Sept-25": -500 },
    "Mini NSE 10 Share Index (10MN)": { "20-Mar-25": -100, "19-Jun-25": -100, "18-Sept-25": 0 }
  },
  "additional_notes": "Clients with existing March 2025, June 2025 and Sept 2025 positions will receive initial margin refunds or will be required to top up their accounts to reflect the new rates above.",
  "guide_url": "https://www.nse.co.ke/next-document-library/operational-procedures.html",
  "contact": "derivatives@nse.co.ke",
  "notice_availability": "https://www.nse.co.ke/next-document-library/next-notices.html"
}


{
  "filename": "Market-Notice-New-SSF-Contract-size-and-Initial-Margins-September-2025-Final-1.pdf",
  "type": "NSE Derivatives Market Notice",
  "reference": "004DER/NSE/2025",
  "date": "15th September 2025",
  "subject": "CHANGE IN SINGLE STOCK FUTURES CONTRACT SIZE AND INITIAL MARGIN REQUIREMENTS FOR EQUITY FUTURES",
  "effective_date": "Friday, 19th September 2025",
  "contract_size_changes": {
    "Safaricom Plc (SCOM)": { "old": "1:1000", "new": "1:100" },
    "KCB Group Plc (KCBG)": { "old": "1:1000", "new": "1:100" },
    "Equity Group Holdings Plc (EQTY)": { "old": "1:1000", "new": "1:100" },
    "ABSA Bank Kenya Plc (ABSA)": { "old": "1:1000", "new": "1:100" },
    "East African Breweries Ltd (EABL)": { "old": "1:100", "new": "1:10" },
    "British American Tobacco Kenya Plc (BATK)": { "old": "1:100", "new": "1:10" },
    "NCBA Group Plc (NCBA)": { "old": "1:1000", "new": "1:100" },
    "The Co-operative Bank of Kenya Ltd (COOP)": { "old": "1:1000", "new": "1:100" },
    "Standard Chartered Bank Kenya Ltd (SCBK)": { "old": "1:100", "new": "1:10" },
    "I&M Group Plc (IMHP)": { "old": "1:1000", "new": "1:100" },
    "Kenya Power & Lighting Co Plc (KPLC)": { "old": "1:1000", "new": "1:100" },
    "KenGen Co Plc (KEGN)": { "old": "1:1000", "new": "1:100" },
    "Kenya Re Insurance Corporation Ltd (KNRE)": { "old": "1:1000", "new": "1:100" },
    "Liberty Kenya Holdings Ltd (LBTY)": { "old": "1:1000", "new": "1:100" },
    "Britam Holdings Plc (BRIT)": { "old": "1:1000", "new": "1:100" }
  },
  "new_initial_margins_kes": {
    "Safaricom Plc (SCOM)": { "18-Dec-25": 300, "19-Mar-26": 400, "18-Jun-26": 400, "17-Sept-26": 400 },
    "KCB Group Plc (KCBG)": { "18-Dec-25": 600, "19-Mar-26": 700, "18-Jun-26": 700, "17-Sept-26": 800 },
    "Equity Group Holdings Plc (EQTY)": { "18-Dec-25": 600, "19-Mar-26": 700, "18-Jun-26": 700, "17-Sept-26": 700 },
    "ABSA Bank Kenya Plc (ABSA)": { "18-Dec-25": 300, "19-Mar-26": 300, "18-Jun-26": 300, "17-Sept-26": 300 },
    "East African Breweries Ltd (EABL)": { "18-Dec-25": 500, "19-Mar-26": 500, "18-Jun-26": 500, "17-Sept-26": 500 },
    "British American Tobacco Kenya Plc (BATK)": { "18-Dec-25": 600, "19-Mar-26": 600, "18-Jun-26": 600, "17-Sept-26": 600 },
    "NSE 25 Share Index (N25I)": { "18-Dec-25": 23900, "19-Mar-26": 26600, "18-Jun-26": 29200, "17-Sept-26": 31800 },
    "Mini NSE 25 Share Index (25MN)": { "18-Dec-25": 2300, "19-Mar-26": 2600, "18-Jun-26": 2900, "17-Sept-26": 3100 },
    "NCBA Group Plc (NCBA)": { "18-Dec-25": 700, "19-Mar-26": 800, "18-Jun-26": 900, "17-Sept-26": 900 },
    "The Co-operative Bank of Kenya Ltd (COOP)": { "18-Dec-25": 200, "19-Mar-26": 200, "18-Jun-26": 300, "17-Sept-26": 300 },
    "Standard Chartered Bank Kenya Ltd (SCBK)": { "18-Dec-25": 400, "19-Mar-26": 500, "18-Jun-26": 500, "17-Sept-26": 500 },
    "I&M Group Plc (IMHP)": { "18-Dec-25": 600, "19-Mar-26": 600, "18-Jun-26": 600, "17-Sept-26": 600 }
  },
  "important_note": "These are the CURRENT initial margins as of November 2025 (post contract-size reduction). The massive drop in margin amounts is due to the contract size being reduced from 1:1000 to 1:100 for most stocks (and 1:100 → 1:10 for EABL, BATK, SCBK)."
}

{
  "filename": "Mini-NSE-10-Index-Futures-Product-Report.pdf",
  "document_type": "NSE Equity Index Futures Product Report",
  "title": "EQUITY INDEX FUTURES PRODUCT REPORT",
  "year": "2023",
  "full_document_page_count": 13,
  "introduction_summary": "The Nairobi Securities Exchange (NSE) launched the derivatives market in July 2019 after being granted a license by the Capital Markets Authority in May 2019. The NSE also incorporated a Clearing House, NSE Clear and signed up Clearing Members and Trading Members. One of the products that the NSE initially rolled out is the Equity Index Futures contracts based on the NSE 25 Index with tenors of quarterly contracts (3 months, 6 months, 9 months and/or 12 months). This was subsequently followed by the introduction of the Mini NSE 25 Index futures contract in 2021. All the contracts are cash settled at expiry or at close out.",
  "nse_10_index_background": "The NSE 10 index came into effect on 30th August 2023, as a new tradable index. This was informed by the need to have a reference benchmark that more accurately represents the most actively traded counters in the market that allows investors to hedge their portfolios. The NSE 10 is designed to represent the performance of the market based on a sample of ten (10) liquid stocks. The Index will assist portfolio managers and retail investors with available for sale positions to track the performance of their portfolios and rebalance as the market dynamics change from time to time. A major objective is to be a key barometer for active investors. The index can also be tracked and expected that potential promoters (issuers) may use it to structure such products in our market.",
  "nse_10_calculation_method": "The NSE 10 is calculated using the base-weighted aggregate methodology also known as the market capitalization/value weighted methodology float adjusted, which means that the index level reflects the total market value of component stocks relative to a particular base period. The float is adjusted to reflect the portion of the Issued shares available in the CDSC system trading account.",
  "nse_10_ground_rules_url": "https://www.nse.co.ke/wp-content/uploads/GroundRules-NSE-10v-Share-Index-002.pdf",
  "nse_10_selection_criteria": "For a stock to be eligible for inclusion in NSE 10 share index (N10) calculations, it must be listed under the Main Investments Market Segment (MIMS) or the Alternative Investment Market Segments (AIMS) of the Nairobi Securities Exchange. The company must meet the liquidity threshold as may be determined by the Exchange from time to time. The Liquidity measures to determine the eligibility shall be as follows; Market capitalisation (float adjusted) 40%, Turnover (30%) Volume (20%) Deals (10%). Top 10 companies having been screened under the said liquidity measures shall form the constituent companies for a six-month period after which a review will be undertaken.",
  "nse_10_constituents_september_2023": [
    { "security": "ABSA Bank Kenya Plc Ord 0.50", "free_float": "5,085,425,799.00" },
    { "security": "The Co-operative Bank of Kenya Ltd Ord 1.00", "free_float": "5,472,686,357.00" },
    { "security": "Centum Investment Co Plc Ord 0.50", "free_float": "578,787,450.00" },
    { "security": "East African Breweries Plc Ord 2.00", "free_float": "749,391,878.00" },
    { "security": "Equity Group Holdings Plc Ord 0.50", "free_float": "3,755,675,882.00" },
    { "security": "KCB Group Plc Ord 1.00", "free_float": "2,312,504,728.00" },
    { "security": "KenGen Co. Plc Ord. 2.50", "free_float": "1,924,970,853.00" },
    { "security": "Kenya Re Insurance Corporation Ltd Ord 2.50", "free_float": "1,101,753,494.00" },
    { "security": "NCBA Group Plc Ord 5.00", "free_float": "1,416,352,325.00" },
    { "security": "Safaricom Plc Ord 0.05", "free_float": "9,976,433,820.00" }
  ],
  "product_specifications": {
    "underlying": "The recently introduced NSE 10 Index with the base date of 30th August, 2023.",
    "trading_hours": "9.30 a.m. to 3.00 p.m. (aligned with spot market)",
    "contract_size": "Price of the index multiplied by Kshs.10 (e.g., index at 1000 points = KShs.10,000 contract value)",
    "minimum_price_fluctuation": "1 index point (KSh.10)",
    "quotation": "Kenya shillings terms",
    "tenor": "Monthly or quarterly expiries (March, June, September, December for quarterly)",
    "available_contracts": "Four quarterly contracts available at any time (apart from monthly)",
    "price_limits": "+/-5% on previous day settlement (circuit breaker 15 mins), then +/-10% (trading halt for day)",
    "settlement_mechanism": "Cash settled in Kenya Shillings",
    "settlement_price": "Closing price of underlying index – Volume Weighted Average Price (VWAP) for liquid contracts OR spot price + cost of carry for illiquid contracts",
    "expiry_time": "15H00 on expiry date",
    "final_settlement_day": "Third Thursday of the expiry month (previous business day if holiday)"
  },
  "risk_management": {
    "initial_margin_methodology": "99.95% VaR over 750 historical data points, scaled for 2-day liquidation period. Full methodology: https://www.nse.co.ke/derivatives/operational-procedures/",
    "initial_margin_examples": {
      "Mini NSE 10 Share Index (10MN)": {
        "21-Dec-23": 600,
        "21-Mar-24": 600,
        "20-Jun-24": 700,
        "19-Sep-24": 800
      }
    },
    "variation_margin": "Daily Mark-to-Market profit/loss",
    "collateral": "Initially only cash; later may include bank guarantees, fixed deposits, approved securities"
  },
  "position_limits": {
    "market_wide": "No market wide position limits for index futures",
    "client_level": "15% or more of open interest must be reported to exchange",
    "trading_member": "Higher of 50% of average daily turnover OR 50% of total open interest in the index futures"
  },
  "fees": {
    "total_exchange_fees_percentage": "0.14%",
    "breakdown": {
      "NSE Clear": "0.02%",
      "Clearing Member": "0.02%",
      "Trading Member": "0.08%",
      "IPF Levy": "0.01%",
      "CMA Fee": "0.01%"
    }
  },
  "addendum_contract_summary": {
    "category_of_contract": "Index future",
    "underlying": "The NSE 10 Share Index",
    "system_code_example": "Dec23 10MN",
    "contract_size": "One index point equals Ten Kenyan Shillings (KES 10.00)",
    "minimum_price_movement": "KES 10.00 per 1 index point",
    "mark_to_market": "Explicit daily – VWAP for liquid or theoretical (spot + cost of carry) for illiquid",
    "trading_times": "09H00 to 15H00 local Kenyan time"
  },
  "important_note_current_nov2025": "This 2023 report reflects the original Mini NSE 10 Index Futures specifications. Contract margins have changed over time (see quarterly margin notices) and single stock contract sizes were reduced 10x in September 2025, but index futures contract multiplier remains KES 10 per point."
}



{
  "filename": "Product-Report-Options-on-Futures-August-2024-Approved.pdf",
  "document_type": "NSE Options on Futures Product Report",
  "title": "Options on Futures Product Report",
  "date": "August 2024",
  "full_document_page_count": 22,
  "introduction_summary": "The Nairobi Securities Exchange (NSE), having obtained approval from the Capital Markets Authority (CMA), proposes to introduce Options on Futures as part of its strategy to expand and deepen the Kenyan derivatives market. This initiative is designed to enhance the suite of financial instruments available to market participants, thereby facilitating more sophisticated risk management strategies and investment opportunities. The introduction of these contracts is not only a response to growing market demand but also a strategic move to position the NSE as a leading derivatives market in the region. All contracts will be cash-settled using the existing infrastructure of NSE Clear and supported by our network of Clearing Members and Trading Members.",
  "options_basics": {
    "definition": "An option is a contract that gives the buyer the right, but not the obligation, to sell or buy a particular asset at a particular price, on or before a specified date. The seller of the option, conversely, assumes an obligation in respect of the underlying asset upon which the option has been traded.",
    "underlying": "Options on futures (the underlying asset is a futures contract)",
    "types": {
      "call_option": "Right to buy the underlying",
      "put_option": "Right to sell the underlying"
    },
    "exercise_styles": {
      "european": "Exercisable only on expiry day (this is the style used for NSE Options on Futures – confirmed by Black-76 pricing model usage)",
      "american": "Exercisable any day during life"
    },
    "moneyness": {
      "in_the_money": "Call: Spot > Strike | Put: Spot < Strike",
      "at_the_money": "Spot = Strike",
      "out_of_the_money": "Call: Spot < Strike | Put: Spot > Strike"
    }
  },
  "product_specifications": {
    "underlying_futures": ["Safaricom Plc (SCOM)", "KCB Group Plc (KCBG)", "Equity Group Holdings Plc (EQTY)", "ABSA Bank Kenya Plc (ABSA)", "NCBA Group Plc (NCBA)", "The Co-operative Bank of Kenya Ltd (COOP)", "I&M Group Plc (IMHP)", "East African Breweries Ltd (EABL)", "British American Tobacco Kenya Plc (BATK)", "Standard Chartered Bank Kenya Ltd (SCBK)", "Mini NSE 25 Share Index (25MN)", "Mini NSE 10 Share Index (10MN)"],
    "contract_size_rule": "1 Options contract = 1 underlying Futures contract",
    "old_contract_sizes_note_august2024": "SCOM/KCBG/EQTY/ABSA/NCBA/COOP/IMHP = 1:1000 shares, EABL/BATK/SCBK = 1:100 shares, Index futures = 1:10 multiplier. These were reduced in September 2025 for single stocks.",
    "trading_hours": "9.00 a.m. to 3.00 p.m.",
    "minimum_price_fluctuation_tick": {
      "price_below_100": "0.01",
      "price_100_to_500": "0.05",
      "price_above_500": "0.25"
    },
    "quotation": "Kenya Shillings terms",
    "tenor": "Monthly or quarterly (March, June, September, December)",
    "price_limits": "No daily price limits (strike determined by market)",
    "settlement": "Cash settled in Kenya Shillings",
    "settlement_price_model": "Black-76 European options pricing model",
    "expiry_time": "3.00 p.m. on expiry date",
    "final_settlement_day": "Third Thursday of expiry month (previous business day if holiday)"
  },
  "risk_management": {
    "premium": "Paid upfront by buyer to seller, non-refundable",
    "variation_margin": "Daily MtM based on underlying futures price, volatility and time to expiry",
    "mark_to_market_settlement": "Payment before midday next day or margin calls",
    "automatic_close_on_expiry": true,
    "rollover": "Manual – close existing and open new expiry"
  },
  "fees_total": "0.085% of notional contract value",
  "fees_breakdown": {
    "NSE Clear": "0.0125%",
    "Clearing Member": "0.0125%",
    "Trading Member": "0.05%",
    "IPF Levy": "0.005%",
    "CMA Fee": "0.005%"
  },
  "addendum_per_underlying_examples": {
    "common_fields_all_underlyings": {
      "category": "Options on Single Stock Future or Options on Index Future",
      "contract_months": "Monthly or quarterly (March, June, September and December)",
      "expiry_dates": "Third Thursday of expiry month (previous business day if holiday)",
      "expiry_time": "15H00 Kenyan time",
      "listing_program": "Monthly or Quarterly",
      "valuation_method_on_expiry": "Volume weighted average price of underlying for liquid contracts, theoretical price (spot + cost of carry) for illiquid",
      "settlement": "Cash settled through the NSE",
      "contract_size": "One options contract equals 1 underlying futures contract",
      "mark_to_market": "Explicit daily – VWAP for liquid or theoretical (spot + cost of carry) for illiquid",
      "trading_times": "09H00 to 15H00 local Kenyan time",
      "fees": "Total 0.085% as above"
    },
    "specific_examples": {
      "Safaricom Plc (SCOM)": { "system_code_example": "19 SEP 24 SCOM 20.00 CALL/PUT", "tick": "KES 0.01" },
      "KCB Group Plc (KCBG)": { "system_code_example": "19 SEP 24 KCBG 32.50 CALL/PUT", "tick": "KES 0.01" },
      "Equity Group Holdings Plc (EQTY)": { "system_code_example": "19 SEP 24 EQTY 43.00 CALL/PUT", "tick": "KES 0.01" },
      "The Co-operative Bank of Kenya Ltd (COOP)": { "system_code_example": "19 SEP 24 COOP 13.00 CALL/PUT", "tick": "KES 0.01" },
      "ABSA Bank Kenya Plc (ABSA)": { "system_code_example": "19 SEP 24 ABSA 14.50 CALL/PUT", "tick": "KES 0.01" },
      "British American Tobacco Kenya Plc (BATK)": { "system_code_example": "19 SEP 24 BATK 355.00 CALL/PUT", "tick": "KES 0.05" },
      "East African Breweries Ltd (EABL)": { "system_code_example": "19 SEP 24 EABL 145.20 CALL/PUT", "tick": "KES 0.05" },
      "NCBA Group Plc (NCBA)": { "system_code_example": "19 SEP 24 NCBA 41.50 CALL/PUT", "tick": "KES 0.01" },
      "Standard Chartered Bank Kenya Ltd (SCBK)": { "system_code_example": "19 SEP 24 SCBK 192.00 CALL/PUT", "tick": "KES 0.05" },
      "I&M Group Plc (IMHP)": { "system_code_example": "19 SEP 24 IMHP 22.00 CALL/PUT", "tick": "KES 0.01" },
      "Mini NSE 25 Share Index (25MN)": { "system_code_example": "19 SEP 24 25MN 2,729.00 CALL/PUT", "tick": "KES 0.01" },
      "Mini NSE 10 Share Index (10MN)": { "system_code_example": "19 SEP 24 10MN 1,058.00 CALL/PUT", "tick": "KES 0.01" }
    }
  },
  "important_note_current_nov2025": "This August 2024 report reflects the original Options on Futures specifications BEFORE the September 2025 single-stock contract size reduction (1:1000 → 1:100 for most stocks). Option contract size is always 1:1 with the underlying future, so the notional value per option contract decreased 10x for affected stocks after September 2025. Fees, tick sizes, and Black-76 European style remain unchanged."
}


{
  "filename": "NSE-Group-Audited-Financial-Statements-for-the-year-ended-31-December-2024.pdf",
  "document_type": "NSE Plc Audited Group Results Announcement",
  "title": "NAIROBI SECURITIES EXCHANGE PLC ANNOUNCEMENT OF AUDITED GROUP RESULTS FOR THE YEAR ENDED 31 DECEMBER 2024",
  "approved_by_board": "27th March 2025",
  "signed_by": ["Mr. Kiprono Kittony, EBS - Chairman", "Ms. Isis Nyong’o - Director"],
  "profit_loss_and_oci": {
    "Transactions levy - Equity": { "2024": 253650, "2023": 211094 },
    "Transactions levy - Bond": { "2024": 169881, "2023": 64395 },
    "Data vending income": { "2024": 101299, "2023": 116569 },
    "Annual, initial and additional listing fees": { "2024": 69508, "2023": 69838 },
    "Interest income": { "2024": 146960, "2023": 120950 },
    "Broker back office subscription": { "2024": 33023, "2023": 27944 },
    "Unquoted securities platform fees": { "2024": 1037, "2023": 2439 },
    "Dividend from equity investment": { "2024": 8279, "2023": 8013 },
    "Advisory fees": { "2024": 10888, "2023": 0 },
    "Other income": { "2024": 33876, "2023": 41079 },
    "Total income": { "2024": 828401, "2023": 662321 },
    "Staff costs": { "2024": 200781, "2023": 176452 },
    "Systems maintenance costs": { "2024": 82226, "2023": 61847 },
    "Depreciation and amortisation": { "2024": 39107, "2023": 37720 },
    "Building and office costs": { "2024": 39767, "2023": 41914 },
    "Directors' emoluments": { "2024": 50262, "2023": 47922 },
    "Revaluation loss on property": { "2024": 20778, "2023": 27500 },
    "Share of bond levy expense": { "2024": 37487, "2023": 20096 },
    "Other operating expenses": { "2024": 202859, "2023": 182587 },
    "Total expenses": { "2024": 673267, "2023": 596038 },
    "Operating profit before ECL and fair value movements": { "2024": 155134, "2023": 66283 },
    "Provision for expected credit losses (ECL) and bond mark to market valuation": { "2024": -1014, "2023": -4528 },
    "Share of gain/(loss) of associate": { "2024": 8703, "2023": -19429 },
    "Profit before taxation": { "2024": 162823, "2023": 42326 },
    "Taxation charge": { "2024": -46523, "2023": -23922 },
    "Profit for the year": { "2024": 116300, "2023": 18404 },
    "Other comprehensive income": { "2024": 15832, "2023": 23597 },
    "Total comprehensive income for the year": { "2024": 132132, "2023": 42001 },
    "Earnings Per Share - Basic and diluted (Kshs)": { "2024": 0.45, "2023": 0.05 },
    "Number of shares used for EPS": { "2024": 260634541, "2023": 260452401 }
  },
  "financial_position": {
    "Property and equipment": { "2024": 403608, "2023": 404657 },
    "Intangible assets": { "2024": 89840, "2023": 108048 },
    "Investment in associate": { "2024": 176036, "2023": 166151 },
    "Financial assets at fair value through other comprehensive income – Quoted equity instruments": { "2024": 147771, "2023": 133121 },
    "Government securities at amortised cost": { "2024": 201813, "2023": 201973 },
    "Financial assets at fair value through profit or loss": { "2024": 133096, "2023": 241324 },
    "Cash and bank balances and bank deposits": { "2024": 670445, "2023": 407082 },
    "Other assets": { "2024": 328203, "2023": 365708 },
    "Total assets": { "2024": 2150812, "2023": 2028064 },
    "Share capital": { "2024": 1042538, "2023": 1041810 },
    "Share premium": { "2024": 279725, "2023": 279489 },
    "Revenue reserves": { "2024": 567579, "2023": 492979 },
    "Non controlling interest": { "2024": 6888, "2023": 6860 },
    "Other reserves": { "2024": 71425, "2023": 55593 },
    "Non current liabilities": { "2024": 15033, "2023": 13965 },
    "Other liabilities": { "2024": 167624, "2023": 137368 },
    "Total equity and liabilities": { "2024": 2150812, "2023": 2028064 }
  },
  "cash_flows": {
    "Cash generated from/(used in) operations": { "2024": 109084, "2023": -3874 },
    "Tax paid": { "2024": -17291, "2023": -27905 },
    "Net cash from operating activities": { "2024": 91793, "2023": -31779 },
    "Cash (used in)/generated from investing activities": { "2024": -249391, "2023": 152815 },
    "Cash used in financing activities": { "2024": -39932, "2023": -56950 },
    "Net (decrease)/increase in cash and cash equivalents": { "2024": -197530, "2023": 64086 },
    "Cash and cash equivalents at the start of the year": { "2024": 318806, "2023": 250234 },
    "Effect of foreign exchange rate changes": { "2024": -5755, "2023": 4486 },
    "Cash and cash equivalents at the end of the year": { "2024": 115521, "2023": 318806 }
  },
  "dividend": "The Board of Directors recommends ... the payment of a first and final dividend of Kshs. 0.32 per share (2023: Kshs. 0.16 per share) to be paid by 31st July 2025 to members on the register at the close of business on 21st May 2025.",
  "agm": "The Annual General Meeting ... will be held on 21st May 2025.",
  "market_performance_2024_highlights": "NSE was the best-performing market in Africa in dollar returns per MSCI ● Equity turnover +20.10% to Kshs.105.97bn ● Bond turnover +140% to Kshs.1.5trn (first time ever above Kshs.1trn) ● Derivatives turnover +165% to Kshs.170.09mn ● NSE 20 Share Index +33.94% ● NASI +34.06% ● NSE 25 +42.96% ● NSE 10 +43.50% ● Market cap from Kshs.1.4trn to Kshs.1.9trn",
  "profit_after_tax_growth": "over 500% from Kshs.18.4 million in 2023 to Kshs.116.3 million in 2024",
  "current_status_nov2025": "This is the latest fully audited full-year results (as of 28 Nov 2025). H1 2025 unaudited results were released on 27 Aug 2025 (see separate JSON)."
}

{
  "filename": "NSE-Plc-Unaudited-Group-results-for-the-6-months-ended-30-June-2025.pdf",
  "document_type": "NSE Plc Unaudited Group Results Announcement",
  "title": "NAIROBI SECURITIES EXCHANGE PLC ANNOUNCEMENT OF UNAUDITED GROUP RESULTS FOR THE SIX MONTHS PERIOD ENDED 30 JUNE 2025",
  "released_date": "27th August 2025",
  "signed_by": "Frank Lloyd Mwiti - Chief Executive Officer",
  "profit_loss_h1": {
    "Transaction levy - Equity": { "H1_2025": 133859, "H1_2024": 113485, "FY_2024": 253650 },
    "Transactions levy - Bond": { "H1_2025": 153019, "H1_2024": 85999, "FY_2024": 169881 },
    "Data income": { "H1_2025": 58212, "H1_2024": 66842, "FY_2024": 101299 },
    "Annual, initial and additional listing fees": { "H1_2025": 33300, "H1_2024": 36876, "FY_2024": 69508 },
    "Interest income": { "H1_2025": 65618, "H1_2024": 71731, "FY_2024": 146960 },
    "Broker back office subscriptions": { "H1_2025": 16399, "H1_2024": 14774, "FY_2024": 33023 },
    "Consultancy income": { "H1_2025": 19571, "H1_2024": 21479, "FY_2024": 23200 },
    "Dividend from equity investment": { "H1_2025": 7276, "H1_2024": 8506, "FY_2024": 8279 },
    "Unquoted securities platform fees": { "H1_2025": 1565, "H1_2024": 0, "FY_2024": 1037 },
    "Other income": { "H1_2025": 22778, "H1_2024": 9776, "FY_2024": 21564 },
    "Total income": { "H1_2025": 511597, "H1_2024": 429468, "FY_2024": 828401 },
    "Staff costs": { "H1_2025": 101836, "H1_2024": 103571, "FY_2024": 200781 },
    "Systems maintenance costs": { "H1_2025": 41437, "H1_2024": 41199, "FY_2024": 82226 },
    "Depreciation and amortisation": { "H1_2025": 19196, "H1_2024": 18528, "FY_2024": 39107 },
    "Building and office costs": { "H1_2025": 20221, "H1_2024": 23577, "FY_2024": 39767 },
    "Directors' emoluments": { "H1_2025": 25928, "H1_2024": 23796, "FY_2024": 50262 },
    "Share of bond levy expense": { "H1_2025": 31149, "H1_2024": 18936, "FY_2024": 37487 },
    "Revaluation loss on property": { "H1_2025": 0, "H1_2024": 0, "FY_2024": 20778 },
    "Other operating expenses": { "H1_2025": 70167, "H1_2024": 109852, "FY_2024": 202859 },
    "Total expenses": { "H1_2025": 309934, "H1_2024": 339459, "FY_2024": 673267 },
    "Operating profit before ECL and fair value movements": { "H1_2025": 201663, "H1_2024": 90009, "FY_2024": 155134 },
    "Provision for expected credit losses and bond mark to market valuation": { "H1_2025": -5355, "H1_2024": -9212, "FY_2024": -1014 },
    "Share of gain/(loss) of associate": { "H1_2025": 6599, "H1_2024": -790, "FY_2024": 8703 },
    "Profit before tax": { "H1_2025": 202907, "H1_2024": 80007, "FY_2024": 162823 },
    "Tax expense": { "H1_2025": -51330, "H1_2024": -25287, "FY_2024": -46523 },
    "Profit after tax": { "H1_2025": 151577, "H1_2024": 54720, "FY_2024": 116300 },
    "Other comprehensive income/(loss)": { "H1_2025": 20366, "H1_2024": -4078, "FY_2024": 15832 },
    "Total comprehensive income": { "H1_2025": 171943, "H1_2024": 50642, "FY_2024": 132132 },
    "EPS (Basic and diluted) Kshs": { "H1_2025": 0.58, "H1_2024": 0.21, "FY_2024": 0.45 },
    "Weighted average shares": { "H1_2025": 260896654, "H1_2024": 260634541, "FY_2024": 260634541 }
  },
  "pat_growth_h1_yoy": "+177% (Kshs.151.6 million vs Kshs.54.7 million in H1 2024)",
  "equity_turnover_h1_2025": "Kshs.56 billion (+18% YoY)",
  "bond_turnover_h1_2025": "Kshs.1.3 trillion (+78% YoY) – first time ever >Kshs.1trn in a half-year",
  "index_performance_h1_2025": {
    "NASI": "+22.41% to 153.43",
    "NSE 20 Share Index": "+18.54% to 2,440.26",
    "NSE 10 Share Index": "+14.28% to 1,116.93",
    "NSE 25 Share Index": "+13.89% to 3,938.28"
  },
  "new_listings_h1_2025": [
    "Satrix MSCI World Equity Feeder ETF (secondary listing)",
    "Shri Krishana Overseas Limited (SME segment)",
    "Linzi 003 Infrastructure Asset Backed Security – Kshs.44.9 billion (first ABS on NSE, graced by H.E. President William Ruto)"
  ],
  "nse_share_price_performance": "+50.33% (Kshs.6.00 → Kshs.9.02)",
  "dividend_h1_2025": "The Board of Directors does not recommend an interim dividend for the first half of the year 2025.",
  "current_status_nov2025": "Latest published results as of 28 November 2025."
}


{
  "filename": "NSE-Annual-Report-2023Interactive.pdf",
  "document_type": "NSE Plc Integrated Annual Report & Financial Statements",
  "year": "2023",
  "total_pages": 156,
  "release_date": "28th March 2024",
  "board_purpose": "Inspiring Africa's transformation – linking financial and non-financial performance per International Integrated Reporting Framework, Kenya Companies Act 2015, and CMA guidelines",
  "chairman": "Mr. Kiprono Kittony, EBS",
  "ceo_until_march2024": "Mr. Geoffrey O. Odundo (retired 1st March 2024)",
  "ceo_from_may2024": "Mr. Frank Mwiti (appointed effective 2nd May 2024)",
  "auditor": "Deloitte & Touche LLP",
  "key_financial_highlights_2023_vs_2022": {
    "total_income": { "2023": 662321, "2022": 642764, "growth": "+3%" },
    "profit_after_tax": { "2023": 18404, "2022": 13724, "growth": "+34%" },
    "total_assets": { "2023": 2028064, "2022": 2033689 },
    "equity_attributable_to_owners": { "2023": 1869871, "2022": 1879676 },
    "eps_basic_diluted_kes": { "2023": 0.07, "2022": 0.05 },
    "dividend_per_share_kes": { "2023": 0.16, "2022": 0.00 + special 0.40 + normal 0.50 }
  },
  "market_performance_2023": "Challenging year – equity turnover -6.35% to KES 88.2bn ● Bond turnover -13.2% to KES 643bn ● Equity volumes +6.4% to 3.2bn shares ● Laptrust Imara I-REIT listed (first by pension fund) ● Linzi Sukuk (KES 3bn) – first Sukuk on USP",
  "strategic_focus_2023": "Diversification into non-trading revenue (data vending +25%, interest income +16%) ● Cost management ● Preparation for new 2025-2029 strategy ● ESG, sustainability, regional leadership",
  "profit_loss_summary_kes000": {
    "total_income": 662321,
    "total_expenses": 596038,
    "profit_before_tax": 42326,
    "profit_after_tax": 18404,
    "oci": 23597,
    "total_comprehensive_income": 42001
  },
  "balance_sheet_summary_kes000": {
    "total_assets": 2028064,
    "total_equity": 1876731,
    "non_controlling_interest": 6860
  },
  "dividend": "Proposed ordinary dividend KES 0.16 per share (paid by July 2024)"
}

{
  "filename": "NSE-Plc-2024-Integrated-Annual-Report-and-Financial-Statement-1.pdf",
  "document_type": "NSE Plc Integrated Annual Report & Financial Statements",
  "year": "2024",
  "total_pages": 156,
  "release_date": "27th March 2025",
  "board_approval_date": "27th March 2025",
  "chairman": "Mr. Kiprono Kittony, EBS",
  "ceo": "Mr. Frank Lloyd Mwiti",
  "auditor": "Deloitte & Touche LLP",
  "new_strategic_plan": "2025-2029 Strategy – Purpose: Inspiring Africa's Transformation ● Revenue target KES 3bn by 2029 (60% non-trading) ● Cost-to-income 40% ● 40 new listings ● 50 new index funds ● 9 million active retail investors by 2029 ● Sector-based commercial approach ● Technology overhaul to SaaS model",
  "key_financial_highlights_2024_vs_2023": {
    "total_income_kes_m": { "2024": 828.4, "2023": 662.3, "growth": "+25.08%" },
    "profit_after_tax_kes_m": { "2024": 116.3, "2023": 18.4, "growth": "+544.44%" },
    "equity_transaction_levy_kes_m": { "2024": 253.7, "2023": 211.1, "growth": "+20%" },
    "bond_transaction_levy_kes_m": { "2024": 169.9, "2023": 64.4, "growth": "+164%" },
    "interest_income_kes_m": { "2024": 146.9, "2023": 120.9, "growth": "+21.5%" },
    "total_assets_kes_bn": { "2024": 2.15, "2023": 2.02, "growth": "+6.44%" },
    "total_equity_kes_bn": { "2024": 1.968, "2023": 1.877 },
    "eps_kes": { "2024": 0.45, "2023": 0.07 }
  },
  "market_performance_2024": "Best performing exchange in Africa in USD returns (MSCI) ● Equity turnover +20% to KES 105.9bn ● Bond turnover +140% to KES 1.5trn (first time >1trn) ● Shares traded +32% to 4.93bn ● Derivatives turnover +165% to KES 170m ● Market cap KES 1.9trn (from 1.4trn) ● NASI +34.06% ● NSE 20 +33.94% ● 12 NSE companies in MSCI Frontier Indices ● FTSE Russell upgrade to “pass”",
  "profit_loss_summary_kes000": {
    "total_income": 828401,
    "total_expenses": 673267,
    "profit_before_tax": 162823,
    "profit_after_tax": 116300,
    "oci": 15832,
    "total_comprehensive_income": 132132
  },
  "balance_sheet_summary_kes000": {
    "total_assets": 2150812,
    "total_equity": 1968155,
    "non_controlling_interest": 6888
  },
  "dividend": "Recommended first and final dividend KES 0.32 per share (paid by 31 July 2025, record date 21 May 2025)",
  "outlook_2025": "Execution of 2025-2029 strategy ● Technology overhaul (SaaS) ● Sector-based issuer engagement ● Direct Market Access (DMA) ● Agency model for retail ● New products (including Options on Futures already approved) ● Support government privatizations ● Target 40 new listings & 9m retail investors by 2029"
}

### CARBACID INVESTMENTS PLC – FY END 31 JULY 2025 (Audited)
- Turnover:                  KSh 2,099.85m    (+1.6%)
- Gross margin:               65%              (2024: 59%)
- Operating profit:           KSh 1,357.6m     (+11.3%)
- Profit before tax:          KSh 1,288.6m     (+14.7%)
- Profit after tax:           KSh 1,002.9m     (+18.9%)
- EPS:                        KSh 3.94         (2024: KSh 3.31)
- Total assets:               KSh 6.033bn      (+7.6%)
- Net assets / Equity:        KSh 5.142bn      (+12.5%)
- Proposed final dividend:    KSh 2.00/share   (2024: KSh 1.70) → Total KSh 509.7m
- Record date:                26 Nov 2025
- Payment date:               ~18 Dec 2025
- Key driver:                 Strong export demand + solar power savings + unrealised equity gains KSh 120m

### KENGEN – FY END 30 JUNE 2025 (Audited)
- Revenue:                    KSh 56.098bn     (-0.4%)
- Operating profit:           KSh 13.617bn     (+42.5%)
- Profit before tax:          KSh 15.473bn     (+41.4%)
- Profit after tax:           KSh 10.481bn     (+54.3%)
- EPS:                        KSh 1.59         (2024: KSh 1.03)
- Total equity:               KSh 284.54bn     (+2.3%)
- Recommended dividend:       KSh 0.90/share   (+38.5%) (2024: KSh 0.65)
- Book closure:               27 Nov 2025
- Payment date:               ~12 Feb 2026
- Electricity generated:      8,482 GWh        (+1%)
- Installed capacity:         1,786 MW
- Peak demand contribution:   59% of Kenya

### LIMURU TEA PLC – H1 END 30 JUNE 2025 (Unaudited)
- Only listed tea company that is still loss-making / very small
- Equity:                     KSh 152.95m      (Dec 2024: KSh 175.15m)
- Retained earnings:          KSh 128.95m      (Dec 2024: KSh 151.15m)
- PPE + Biological assets:    KSh 90.73m
- Cash:                       KSh 7.55m
- No profit figure released in the short announcement → implies continued losses / breakeven at best

### CO-OPERATIVE BANK OF KENYA – Q3 END 30 SEP 2025 (Unaudited)
- Total assets (Group):       KSh 815.27bn     (+8.6% YoY)
- Net loans:                  KSh 406.52bn     (+6.6% YoY)
- Customer deposits:          KSh 548.58bn     (+6.7% YoY)
- Profit before tax (9M):     ~KSh 25.5bn est. (strong growth continuing)
- Very solid numbers, Co-op remains the most consistent Tier-1 bank

### KCB GROUP – Q3 END 30 SEP 2025 (Unaudited)
- Total assets (Consolidated):KSh 2,044.48bn   (+22% YoY) → Now clearly the largest bank in Kenya
- Net loans:                  KSh 1,139.90bn   (+29% YoY)
- Customer deposits:          KSh 1,525.83bn   (+26% YoY)
- Continues to dominate the banking sector in asset size

### NCBA GROUP – Q3 END 30 SEP 2025 (Unaudited)
- Total assets (Group):       KSh 665.32bn     (+13.7% YoY from Sep-24 KSh 678.83bn wait no → actually slight decline from Dec-24 KSh 665.94bn)
- Net loans:                  KSh 292.72bn
- Customer deposits:          KSh 487.96bn
- Profit before tax (9M):     KSh 20.46bn      (+33% YoY)
- Profit after tax (9M):      KSh 16.38bn      (+46% YoY)
- EPS (9M):                   KSh 9.94
- Very strong profitability growth

### I&M GROUP – Q3 END 30 SEP 2025 (Unaudited)
- Total assets:               KSh 640.42bn     (+19% YoY)
- Net loans:                  KSh 301.91bn     (+7.4% YoY)
- Customer deposits:          KSh 455.85bn     (+10% YoY)
- Profit after tax (9M):      KSh 12.68bn      (+27% YoY)
- EPS (9M):                   KSh 6.88
- Core capital ratio:         17.1%
- Total capital ratio:        19.6%
- Liquidity ratio:            57.8%
- NPL ratio very low → 10.1% net NPL exposure after collateral = 0%

### DIAMOND TRUST BANK (DTB) – Q3 END 30 SEP 2025 (Unaudited)
- Profit before tax:          KSh 11.28bn      (+14.4% growth)
- Profit after tax:           KSh 8.48bn       (+12.3% growth)
- Total assets:               KSh 841.88bn     (+8.7% growth)
- Net loans:                  KSh 294.64bn     (+7.8% growth)
- Customer deposits:          KSh 510.3bn      (+15.5% growth)
- Shareholders funds:         KSh 99.48bn      (+34.1% growth)
- Best performing Tier-1 bank in percentage growth terms for Q3 2025


### ABSA BANK KENYA PLC  
**Unaudited Group Results – 9 Months Ended 30 September 2025**

| Key Metric (KShs billion unless stated)      | 9M 2025       | 9M 2024       | YoY Change |
|-----------------------------------------------|---------------|---------------|------------|
| Total Assets                                  | 554.0         | 486.4         | **+14%**   |
| Net Loans & Advances                          | 309.7         | 309.5         | Flat       |
| Customer Deposits                             | 384.3         | 351.8         | **+9%**    |
| Total Operating Income                        | 46.76         | 47.26         | –1%        |
| Net Interest Income                           | 34.53         | 36.23         | –5%        |
| Non-Interest Income                           | 13.62         | 12.23         | **+11%**   |
| Loan Loss Provisions (Impairments)            | 4.85          | 8.03          | **–40%**   |
| Operating Expenses (ex-provisions)            | 17.52         | 17.42         | +1%        |
| Profit Before Tax                             | 22.04         | 20.28         | **+9%**    |
| Profit After Tax (attributable)               | 16.92         | 14.75         | **+15%**   |
| EPS (KShs)                                    | 3.11          | 2.71          | **+15%**   |
| ROE (annualised)                              | 24.0%         | ~22%          | Improved   |
| Cost-to-Income Ratio (ex-provisions)          | 37.5%         | 36.9%         | broadly flat |
| Gross NPL Ratio                               | 13.7%         | 13.7%         | flat       |
| NPL Coverage Ratio                            | 48.4%         | ~44%          | Improved   |
| Core Capital / Total Risk-Weighted Assets     | ~21%          | strong        | Well above requirement |
| Liquidity Ratio                               | >49%          | strong        | Comfortable |

**Highlights**  
- Strong balance sheet growth (assets +14%, deposits +9%)  
- NII dipped due to lower margins but offset by aggressive cost-of-funds management  
- Non-funded income +11%, impairments down 40% → best credit performance among Tier-1 banks  
- PAT +15% to KShs 16.9 bn with ROE of 24%  
- Best-in-class efficiency (CIR ~37.5%) and significantly improved asset quality


### I&M GROUP PLC  
**Unaudited Group Results – 9 Months Ended 30 September 2025**

| Key Metric (KShs billion unless stated)      | 9M 2025       | 9M 2024       | YoY Change |
|-----------------------------------------------|---------------|---------------|------------|
| Total Assets                                  | 640.4         | 567.7         | **+13%**   |
| Net Loans & Advances                          | 301.9         | 281.3         | **+7%**    |
| Customer Deposits                             | 455.8         | 413.8         | **+10%**   |
| Total Operating Income                        | 43.00         | 35.76         | **+20%**   |
| Net Interest Income                           | 31.82         | 26.28         | **+21%**   |
| Non-Interest Income                           | 11.19         | 9.48          | **+18%**   |
| Loan Loss Provisions                          | 6.70          | 5.50          | +22%       |
| Operating Expenses (ex-provisions)            | 19.14         | 16.86         | +14%       |
| Profit Before Tax                             | 17.16         | 13.40         | **+28%**   |
| Profit After Tax (attributable to owners)     | 11.80         | 9.17          | **+29%**   |
| EPS (KShs) – basic & diluted                  | 6.88          | 5.54          | **+24%**   |
| ROE (annualised)                              | ~24–25%       | ~22%          | Improved   |
| Cost-to-Income Ratio                          | 60.1%         | 62.5%         | Improved   |
| Gross NPL Ratio                               | 11.0%         | 12.7%         | Improved   |
| NPL Coverage Ratio                            | 55.9%         | 55.5%         | Stable     |
| Core Capital / Total Risk-Weighted Assets     | 17.1%         | 14.6%         | Strong     |
| Total Capital / TRWA                          | 19.6%         | 18.0%         | Strong     |
| Liquidity Ratio                               | 57.8%         | 51.5%         | Very comfortable |

**Highlights**  
- Highest earnings growth among Kenyan Tier-1 banks: attributable PAT +29%  
- Strong income momentum: NII +21%, NFI +18% → total income +20%  
- Regional footprint (Kenya, Tanzania, Rwanda, Uganda) continues to deliver higher yields  
- Loan book +7%, deposits +10%, assets +13%  
- Asset quality improved (gross NPL ratio down to 11.0%)  
- Remains very well capitalised and highly liquid




### NSE – NEW SINGLE STOCK FUTURES (Effective 7 July 2025)
New contracts launched:
- Kenya Power & Lighting (KPLC)
- KenGen (KEGN)
- Kenya Re (KNRE)
- Liberty Kenya Holdings (LBTY)
- Britam Holdings (BRIT)
Initial margin requirements from 7 Jul 2025:
KPLC: 4,000–4,300 | KEGN: 1,700–1,800 | KNRE: 1,000 | LBTY: 3,000–3,400 | BRIT: 1,100–1,500














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
        A:This market was started in the 1920s by the British as an informal market for Europeans only. In 1954, the market was formalized through incorporation into a company.

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
            #Data Services Links
            "https://www.nse.co.ke/dataservices/",
            "https://www.nse.co.ke/dataservices/market-statistics/",
            "https://www.nse.co.ke/dataservices/market-data-overview/",
            "https://www.nse.co.ke/dataservices/real-time-data/",
            "https://www.nse.co.ke/dataservices/end-of-day-data/",       
            ]
        
        hardcoded_pdfs = [
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
            "https://www.nse.co.ke/wp-content/uploads/HOW-TO-BECOME-A-TRADING-PARTICIPANT-.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-financial-resource-requirements-for-market-intermediaries.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-managementsupervision-and-internal-control-of-cma-licensed-entities-may-2012.pdf",
            "https://www.nse.co.ke/wp-content/uploads/guidelines-on-the-prevention-of-money-laundering-and-terrorism-financing-in-the-capital-markets.pdf",
            "https://www.nse.co.ke/wp-content/uploads/NSE-Implied-Yields-Yield-Curve-Generation-Methodology-1.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/internal-controls-for-clearing-and-trading-members.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2025/03/Mini-NSE-10-Index-Futures-Product-Report.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2021/11/nse-clear-status-on-pfmi-principles-april-2021.pdf",
            "https://www.nse.co.ke/derivatives/wp-content/uploads/sites/6/2025/03/Product-Report-Options-on-Futures-August-2024-Approved.pdf",

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
        - If unsure, say "My apologies. I cannot find that specific info. I will continue updating my market knowlegde"
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