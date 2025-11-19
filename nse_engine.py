import os
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# UPDATED IMPORT: Use the specific text splitters package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# UPDATED IMPORT: Use langchain_core for documents
from langchain_core.documents import Document
from langchain.chains import RetrievalQA

class NSEKnowledgeBase:
    def __init__(self, openai_api_key):
        """
        Initialize the bot engine with OpenAI.
        """
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required")
            
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # 1. Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            temperature=0.0
        )
        
        # 2. Initialize OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # 3. Initialize Vector Store
        # We use a persistent directory so we don't lose data on restart
        self.db_directory = "./nse_db"
        self.vector_db = Chroma(
            persist_directory=self.db_directory, 
            embedding_function=self.embeddings
        )
        
        self.qa_chain = None

    def scrape_nse_website(self, urls):
        """
        Scrapes text content from a list of NSE URLs.
        """
        documents = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        status_logs = []

        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer"]):
                        script.decompose()
                        
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    clean_text = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    documents.append(Document(page_content=clean_text, metadata={"source": url}))
                    status_logs.append(f"✅ Scraped: {url}")
                else:
                    status_logs.append(f"❌ Failed {url}: Status {response.status_code}")
            except Exception as e:
                status_logs.append(f"❌ Error {url}: {str(e)}")
                
        return documents, status_logs

    def build_knowledge_base(self, urls=None):
        """
        Main function to update the bot's knowledge.
        """
        if not urls:
            urls = [
                "https://www.nse.co.ke/",
                "https://www.nse.co.ke/market-statistics/",
                "https://www.nse.co.ke/listed-companies/",
                "https://www.nse.co.ke/market-reports/"
            ]

        raw_docs, logs = self.scrape_nse_website(urls)
        
        if not raw_docs:
            return "No data found.", logs

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(raw_docs)

        # Add to Vector DB
        self.vector_db.add_documents(docs)
        
        return f"Success! Indexed {len(docs)} chunks of data.", logs

    def _init_chain(self):
        """Initializes the Q&A chain"""
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def answer_question(self, query):
        """
        The main function your bot calls.
        """
        if not self.qa_chain:
            self._init_chain()
            
        try:
            result = self.qa_chain.invoke({"query": query})
            answer = result["result"]
            
            # Extract unique sources
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]]))
            
            return answer, sources
        except Exception as e:
            return f"Error processing request: {str(e)}", []