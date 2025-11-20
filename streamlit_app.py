# --- CRITICAL FIX FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import streamlit as st
from nse_engine import NSEKnowledgeBase
import threading
import time

# Minimalist Configuration
st.set_page_config(page_title="NSE Assistant", page_icon="💬", layout="centered")

# --- CUSTOM CSS FOR ADAPTIVE THEME ---
st.markdown("""
    <style>
    .stChatInput { border-radius: 20px; }
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 20px;
        text-align: center;
        color: var(--text-color); 
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">NSE Digital Assistant</div>', unsafe_allow_html=True)

# --- AUTHENTICATION ---
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.stop()

@st.cache_resource
def get_nse_engine(api_key):
    return NSEKnowledgeBase(openai_api_key=api_key)

try:
    nse_bot = get_nse_engine(openai_api_key)
except Exception as e:
    st.error("Service temporarily unavailable.")
    st.stop()

# --- BACKGROUND UPDATE ---
if "update_thread_started" not in st.session_state:
    st.session_state["update_thread_started"] = False

last_update_ts = nse_bot.get_last_update_time()
current_ts = time.time()
hours_since_update = (current_ts - last_update_ts) / 3600

if not nse_bot.has_data() or hours_since_update > 24:
    if not st.session_state["update_thread_started"]:
        def run_background_scrape():
            nse_bot.build_knowledge_base()
        
        t = threading.Thread(target=run_background_scrape)
        t.start()
        st.session_state["update_thread_started"] = True
        st.toast("System is updating NSE data in the background...", icon="🔄")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm the NSE Digital Assistant."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            stream, sources = nse_bot.answer_question(prompt)
            
            if isinstance(stream, str):
                st.markdown(stream)
                st.session_state.messages.append({"role": "assistant", "content": stream})
            else:
                response = st.write_stream(stream)
                
                if sources:
                    source_text = "\n\n**Sources:** \n" + "  \n".join([f"• [{s.replace('https://www.nse.co.ke', 'nse.co.ke')}]({s})" for s in sources])
                    st.markdown(source_text) 
                    response += source_text

                st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception:
            err_msg = "I'm having trouble connecting to the market data. Please try again."
            st.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})

with st.sidebar:
    if st.button("Clear Chat", type="primary"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm the NSE Digital Assistant. Ask me about trading rules, board members, or listed companies."}]
        st.rerun()