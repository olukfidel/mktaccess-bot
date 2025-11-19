# --- CRITICAL FIX FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import streamlit as st
from nse_engine import NSEKnowledgeBase

st.set_page_config(page_title="NSE Smart Chatbot", page_icon="📈")
st.title("📈 NSE Context-Aware Chatbot")

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

with st.sidebar:
    st.header("🧠 Knowledge Base")
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
            if sources:
                st.markdown(f"\n\n**Sources:** {', '.join(sources)}")
            
            full_response = f"{answer}\n\n**Sources:** {', '.join(sources)}" if sources else answer
            st.session_state.messages.append({"role": "assistant", "content": full_response})