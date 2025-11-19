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
# 1. Try to get key from Streamlit Secrets (Best for Cloud)
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
# 2. If not in secrets, ask the user (Best for Local Testing)
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

with st.sidebar:
    st.header("🧠 Knowledge Base")
    # Only show the Update button if we are running locally or if you specifically want users to update it
    # You can hide this from normal users if you want by checking for a specific password
    if st.button("🔄 Update NSE Data"):
        with st.spinner("Scraping..."):
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
                # formatted_sources = "\n\n**Sources:** " + ", ".join(sources)
                # st.markdown(formatted_sources)
                # full_response += formatted_sources
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.write(f"- {source}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})