import os
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# We strictly use the modern chain syntax now
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class NSEKnowledgeBase:
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required")
            
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
        self.embeddings = OpenAIEmbeddings()
        
        self.db_directory = "./nse_db"
        self.vector_db = Chroma(
            persist_directory=self.db_directory, 
            embedding_function=self.embeddings
        )
        
        self.qa_chain = None

    def scrape_nse_website(self, urls):
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
                    for script in soup(["script", "style", "nav", "footer"]):
                        script.decompose()
                    text = soup.get_text()
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(raw_docs)
        self.vector_db.add_documents(docs)
        return f"Success! Indexed {len(docs)} chunks of data.", logs

    def _init_chain(self):
        # Initialize using modern create_retrieval_chain
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
        
        system_prompt = (
            "You are a helpful assistant for the Nairobi Securities Exchange (NSE). "
            "Use the following pieces of retrieved context to answer the user's question. "
            "If the answer is not in the context, politely say you don't have that information. "
            "\n\nContext:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    def answer_question(self, query):
        if not self.qa_chain:
            self._init_chain()
            
        try:
            result = self.qa_chain.invoke({"input": query})
            answer = result["answer"]
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in result.get("context", [])]))
            
            return answer, sources
        except Exception as e:
            return f"Error processing request: {str(e)}", []