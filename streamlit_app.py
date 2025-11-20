# --- CRITICAL FIX FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import streamlit as st
from nse_engine import NSEKnowledgeBase
import threading
import time
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="NSE Investor Assistant", page_icon="📊", layout="wide")

# Custom CSS for Enterprise Look
st.markdown("""
    <style>
    .stChatInput {border-radius: 15px;}
    .main-header {font-size: 2.5rem; font-weight: 700; color: #0F4C81;}
    .sub-header {font-size: 1.1rem; color: #666;}
    div.stButton > button:first-child {border-radius: 20px;}
    </style>
""", unsafe_allow_html=True)

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
    st.error(f"System Error: {e}")
    st.stop()

# --- ANALYTICS LOGGER ---
def log_query(query, feedback=None):
    if "query_log" not in st.session_state:
        st.session_state.query_log = []
    
    entry = {"timestamp": datetime.now().strftime("%H:%M:%S"), "query": query, "feedback": feedback}
    st.session_state.query_log.append(entry)

# --- BACKGROUND UPDATE MANAGER ---
if "update_thread_started" not in st.session_state:
    st.session_state["update_thread_started"] = False

last_update_ts = nse_bot.get_last_update_time()
current_ts = time.time()
hours_since_update = (current_ts - last_update_ts) / 3600

if not nse_bot.has_data() or hours_since_update > 6:
    if not st.session_state["update_thread_started"]:
        def run_background_scrape():
            nse_bot.build_knowledge_base()
        
        t = threading.Thread(target=run_background_scrape)
        t.start()
        st.session_state["update_thread_started"] = True
        
        if not nse_bot.has_data():
            st.toast("⚙️ System initializing... Gathering market data.", icon="📡")

# --- LAYOUT ---
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="main-header">📊 NSE Investor Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enterprise-Grade Market Intelligence</div>', unsafe_allow_html=True)

    # --- CHAT INTERFACE ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your NSE Financial Analyst. Ask me about market trends, specific companies, or trading rules."}]

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add Feedback Buttons for Assistant messages
            if message["role"] == "assistant" and i > 0:
                c1, c2, c3 = st.columns([1, 1, 10])
                with c1:
                    if st.button("👍", key=f"up_{i}"):
                        st.toast("Thanks for the feedback!")
                        # In a real app, save this to DB
                with c2:
                    if st.button("👎", key=f"down_{i}"):
                        st.toast("We'll improve this answer.")

    if prompt := st.chat_input("E.g., 'What is the current price of Safaricom?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        log_query(prompt) # Log for analytics
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                stream, sources = nse_bot.answer_question(prompt)
                
                if isinstance(stream, str): # Handle refusal/error
                    st.warning(stream)
                    st.session_state.messages.append({"role": "assistant", "content": stream})
                else:
                    response = st.write_stream(stream)
                    
                    if sources:
                        with st.expander("📚 Verified Sources"):
                            for source in sources:
                                st.markdown(f"- [{source}]({source})")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Follow-up Suggestions (Simple logic based on keywords)
                    st.markdown("**Suggested Follow-ups:**")
                    f1, f2 = st.columns(2)
                    if "price" in prompt.lower():
                        with f1: st.button("📈 Show historical performance")
                        with f2: st.button("📰 Latest news for this company")
                    elif "who" in prompt.lower():
                        with f1: st.button("🏢 View Board of Directors")
                        with f2: st.button("📞 Contact Information")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

with col2:
    st.image("https://www.nse.co.ke/wp-content/uploads/2019/05/NSE-Logo.png", width=150)
    st.markdown("### System Status")
    
    if last_update_ts > 0:
        st.success(f"**Data Live**\nUpdated: {time.ctime(last_update_ts)}")
    else:
        st.warning("Data Synchronizing...")
    
    st.info("This bot uses a Hybrid Search Engine to combine keyword accuracy with semantic understanding.")
    
    with st.expander("📊 Analytics Dashboard"):
        if "query_log" in st.session_state and st.session_state.query_log:
            df = pd.DataFrame(st.session_state.query_log)
            st.dataframe(df[["timestamp", "query"]], hide_index=True)
            st.caption(f"Total Queries: {len(df)}")
        else:
            st.caption("No queries yet.")
            
    st.markdown("---")
    st.caption("© 2025 Market Access Bot.\n STRICTLY CONFIDENTIAL.")