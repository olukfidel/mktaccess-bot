# --- CRITICAL FIX FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import streamlit as st
from nse_engine import NSEKnowledgeBase
import threading
import time

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

# --- SILENT BACKGROUND AUTO-UPDATE LOGIC ---
if "update_thread_started" not in st.session_state:
    st.session_state["update_thread_started"] = False

# Check last update time
last_update_ts = nse_bot.get_last_update_time()
current_ts = time.time()
hours_since_update = (current_ts - last_update_ts) / 3600

# Condition: If no data exists OR data is older than 6 hours
if not nse_bot.has_data() or hours_since_update > 6:
    if not st.session_state["update_thread_started"]:
        
        def run_background_scrape():
            # This runs in a separate thread
            print("🔄 Starting background NSE scrape...")
            nse_bot.build_knowledge_base()
            print("✅ Background scrape finished.")
        
        # Start the thread
        t = threading.Thread(target=run_background_scrape)
        t.start()
        
        st.session_state["update_thread_started"] = True
        
        # Notify user non-intrusively
        if not nse_bot.has_data():
            st.info("⚙️ Initial setup: Creating knowledge base... (Chat may be slow for 15 seconds)")
        else:
            st.toast("Refreshing market data in the background...", icon="🔄")
# -------------------------------------------

# Sidebar is now purely informational (No Button)
with st.sidebar:
    st.header("🧠 Knowledge Base")
    if last_update_ts > 0:
        st.write(f"Last Updated: {time.ctime(last_update_ts)}")
    else:
        st.write("Status: Initializing...")
    
    st.info("Data updates automatically every 6 hours.")

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