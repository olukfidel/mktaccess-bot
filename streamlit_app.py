# --- CRITICAL FIX FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import streamlit as st
from nse_engine import NSEKnowledgeBase
import threading
import time
import base64

# --- CONFIGURATION ---
st.set_page_config(page_title="NSE Digital Assistant", page_icon="📈", layout="centered")

# --- HELPER: LOAD IMAGE AS BASE64 ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

logo_base64 = get_base64_of_bin_file("logo.webp")

# --- THEME TOGGLE STATE ---
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# --- TOGGLE BUTTON IN SIDEBAR ---
with st.sidebar:
    st.image("https://i.postimg.cc/NF1qzmFV/nse-small-logo.png", width=100) 
    if st.button("Toggle Dark/Light Mode"):
        if st.session_state.theme == "light":
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"
        st.rerun()

# --- DYNAMIC CSS (THEME AWARE) ---
if st.session_state.theme == "light":
    # Light Mode Colors
    text_color = "#333333"
    bg_overlay_color = "rgba(255, 255, 255, 0.85)"
    chat_bg = "#ffffff"
    user_border = "#0F4C81"
    bot_border = "#4CAF50"
    header_color = "#0F4C81"
    input_bg = "#ffffff"
    input_text = "#333333"
else:
    # Dark Mode Colors
    text_color = "#f0f0f0"
    bg_overlay_color = "rgba(20, 20, 20, 0.9)" # Darker overlay for contrast
    chat_bg = "#2b2b2b"
    user_border = "#4da6ff"
    bot_border = "#81c784"
    header_color = "#ffffff"
    input_bg = "#2b2b2b"
    input_text = "#ffffff"

# Construct CSS with explicit colors to override browser defaults
st.markdown(f"""
    <style>
    /* Force text color on all elements to match our theme */
    html, body, [class*="css"], .stMarkdown, .stText {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: {text_color} !important; 
    }}
    
    /* Background Image with Overlay */
    .stApp {{
        background-image: url("data:image/webp;base64,{logo_base64}");
        background-size: 50%;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        background-color: {bg_overlay_color}; 
        background-blend-mode: overlay;
    }}

    /* Header Styling */
    .main-header {{
        color: {header_color} !important;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .sub-header {{
        color: {text_color} !important;
        opacity: 0.8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }}

    /* Chat Input Styling */
    .stChatInput textarea {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border-radius: 20px !important;
    }}
    
    .stChatInputContainer {{
        border-radius: 25px !important;
        border: 2px solid {header_color} !important;
        background-color: {input_bg} !important;
    }}

    /* Chat Message Bubbles */
    .stChatMessage {{
        background-color: {chat_bg} !important;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }}
    
    /* Message Text Color Inside Bubbles */
    .stChatMessage p {{
        color: {text_color} !important;
    }}

    /* Borders for User vs Assistant */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {{
        border-left: 5px solid {user_border};
    }}
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {{
        border-left: 5px solid {bot_border};
    }}

    /* Hide Standard Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Fix Link Colors */
    a {{
        color: {user_border} !important;
        text-decoration: none;
    }}
    a:hover {{
        text-decoration: underline;
    }}
    </style>
""", unsafe_allow_html=True)

# --- UI HEADER ---
st.markdown('<div class="main-header">Nairobi Securities Exchange</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Digital Assistant & Market Intelligence</div>', unsafe_allow_html=True)

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
    st.error("System initializing... Please wait.")
    st.stop()

# --- BACKGROUND UPDATE ---
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

# --- CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the NSE Digital Assistant. You can ask me about share prices, trading rules, or market reports."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="https://i.postimg.cc/NF1qzmFV/nse-small-logo.png" if message["role"] == "assistant" else None):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the market..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="https://i.postimg.cc/NF1qzmFV/nse-small-logo.png"):
        try:
            stream, sources = nse_bot.answer_question(prompt)
            
            if isinstance(stream, str):
                st.markdown(stream)
                st.session_state.messages.append({"role": "assistant", "content": stream})
            else:
                response = st.write_stream(stream)
                if sources:
                    source_text = "\n\n**Sources:** \n" + "  \n".join([f"• [{s.replace('https://www.nse.co.ke', 'nse.co.ke').split('/')[-1]}]({s})" for s in sources])
                    st.markdown(source_text)
                    response += source_text
                
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception:
            err = "I am currently synchronizing with the latest market data. Please try again in a moment."
            st.warning(err)