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
# We use CSS variables (var(--...)) so it adapts to light/dark mode automatically.
st.markdown("""
    <style>
    /* Chat Input Styling */
    .stChatInput {
        border-radius: 20px;
    }
    
    /* Main Header Styling */
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 20px;
        text-align: center;
        /* Uses Streamlit's primary text color variable */
        color: var(--text-color); 
    }
    
    /* Hide the default Streamlit hamburger menu and footer for a cleaner look */
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

# --- INVISIBLE BACKGROUND UPDATE ---
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

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi there! I'm your NSE assistant. How can I help you today?"}]

# Display chat history cleanly
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input logic
if prompt := st.chat_input("Type your message..."):
    # 1. Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 2. Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate Response
    with st.chat_message("assistant"):
        try:
            stream, sources = nse_bot.answer_question(prompt)
            
            if isinstance(stream, str): # Error handling
                st.markdown(stream)
                st.session_state.messages.append({"role": "assistant", "content": stream})
            else:
                response = st.write_stream(stream)
                
                # Append sources unobtrusively at the bottom of the response
                if sources:
                    # We create a clean, small footnote for sources
                    source_text = "\n\n**Sources:** \n" + "  \n".join([f"• [{s.replace('https://www.nse.co.ke', 'nse.co.ke')}]({s})" for s in sources])
                    st.markdown(source_text) 
                    response += source_text  # Add to history so it persists

                st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception:
            err_msg = "I'm having trouble connecting to the market data. Please try again."
            st.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})

# Add a subtle reset button in the sidebar (optional, keeps main UI clean)
with st.sidebar:
    if st.button("Clear Chat", type="secondary"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! I'm your NSE assistant. How can I help you today?"}]
        st.rerun()