# --- CRITICAL FIX FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import streamlit as st
from nse_engine import NSEKnowledgeBase

st.set_page_config(page_title="NSE Smart Chatbot", page_icon="📈")
st.title("📈 NSE Context-Aware Chatbot")

# --- AUTHENTICATION ---
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

@st.cache_resource
def get_nse_engine(api_key):
    return NSEKnowledgeBase(openai_api_key=api_key)

try:
    nse_bot = get_nse_engine(openai_api_key)
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- OPTIMIZED STARTUP LOGIC ---
# We check if we have scraped in this session OR if the DB already has data.
if "startup_check_done" not in st.session_state:
    with st.spinner("Checking knowledge base..."):
        try:
            # Optimization: Only scrape if the database is empty
            if not nse_bot.has_data():
                st.write("🚀 Database empty. Performing initial scrape...")
                status_msg, logs = nse_bot.build_knowledge_base()
                st.success(status_msg)
            else:
                # Data exists, no need to scrape!
                st.toast("✅ Loaded data from cache (Fast Start)", icon="⚡")
            
            st.session_state["startup_check_done"] = True
            
        except Exception as e:
            st.error(f"Startup failed: {e}")
# -------------------------------

with st.sidebar:
    st.header("🧠 Knowledge Base")
    
    # Manual override to force a refresh
    if st.button("⚠️ Force Re-Scrape"):
        with st.spinner("Scraping 20+ pages in parallel..."):
            status_msg, logs = nse_bot.build_knowledge_base()
            st.success(status_msg)
            with st.expander("View Scraping Logs"):
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