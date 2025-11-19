# --- CRITICAL FIX FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import streamlit as st
from nse_engine import NSEKnowledgeBase

st.set_page_config(page_title="NSE Smart Chatbot", page_icon="📈")
st.title("📈 NSE Context-Aware Chatbot")

# --- MODIFIED AUTHENTICATION LOGIC ---
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()
# -------------------------------------

@st.cache_resource
def get_nse_engine(api_key):
    return NSEKnowledgeBase(openai_api_key=api_key)

try:
    nse_bot = get_nse_engine(openai_api_key)
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- AUTO SCRAPE ON STARTUP LOGIC ---
# This checks if we have run the scrape in this session yet.
if "auto_scraped" not in st.session_state:
    with st.spinner("🚀 Performing initial NSE market data scrape... (This runs once on startup)"):
        try:
            status_msg, logs = nse_bot.build_knowledge_base()
            st.session_state["auto_scraped"] = True
            st.success("Startup Data Refresh Complete! " + status_msg)
            
            # Optional: Log details to console or expander if you debugging
            # with st.expander("Startup Logs"):
            #    for log in logs: st.write(log)
            
        except Exception as e:
            st.error(f"Automatic scraping failed: {e}")
# ------------------------------------

with st.sidebar:
    st.header("🧠 Knowledge Base")
    st.write("Data is refreshed automatically on startup.")
    
    # Kept as a manual override in case data changes while the user is using the app
    if st.button("⚠️ Force Re-Scrape"):
        with st.spinner("Forcing fresh scrape..."):
            status_msg, logs = nse_bot.build_knowledge_base()
            st.success(status_msg)
            with st.expander("Logs"):
                for log in logs:
                    st.text(log)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about NSE..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = nse_bot.answer_question(prompt)
            st.markdown(answer)
            
            full_response = answer
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.write(f"- {source}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})