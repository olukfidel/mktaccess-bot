# --- CRITICAL FIX FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import streamlit as st
import requests
import time
import base64
import os
import threading

# --- CONFIGURATION ---
st.set_page_config(page_title="NSE Digital Assistant", page_icon="https://i.postimg.cc/NF1qzmFV/nse_small_logo.png", layout="centered")

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

# --- SILENT AUTHENTICATION ---
# Try to get key from secrets. If not there, show error (but don't ask user).
if "OPENAI_API_KEY" in st.secrets:
    # We don't need to do anything with it here if using the API backend,
    # but it's good practice to check it exists.
    pass 
else:
    st.error("System Configuration Error: API Key missing in secrets.")
    st.stop()

# --- API CONFIGURATION ---
# Get URL from secrets or default to local for testing
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# --- BACKGROUND SCRAPER ---
# This runs ONCE when the app script first executes, but outside the Streamlit reruns
if "startup_scrape_triggered" not in st.session_state:
    st.session_state.startup_scrape_triggered = True
    
    def silent_refresh():
        """Calls the backend refresh endpoint without blocking the UI."""
        try:
            # We add a small delay to ensure backend is ready if it's a cold start
            time.sleep(2) 
            requests.post(f"{API_URL}/refresh", timeout=1) # Short timeout, we don't wait for result
        except:
            pass # Fail silently if backend is unreachable (user will see error on chat attempt)

    # Launch the thread
    t = threading.Thread(target=silent_refresh)
    t.start()
    
    # Optional: Show a small toast to the first user
    st.toast("System initializing... syncing market data.", icon="🔄")


# --- TOGGLE BUTTON IN SIDEBAR ---
with st.sidebar:
    st.image("https://i.postimg.cc/NF1qzmFV/nse-small-logo.png", width=100) 
    if st.button("Toggle Dark/Light Mode"):
        if st.session_state.theme == "light":
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"
        st.rerun()
        
    # We keep this manual button hidden or small for admins, or remove it entirely
    # st.markdown("---")
    # if st.button("Force Refresh"): ...

# --- DYNAMIC CSS (THEME AWARE) ---
if st.session_state.theme == "light":
    text_color = "#333333"
    bg_overlay_color = "rgba(255, 255, 255, 0.85)"
    chat_bg = "#ffffff"
    user_border = "#0F4C81"
    bot_border = "#4CAF50"
    header_color = "#0F4C81"
    input_bg = "#ffffff"
    input_text = "#333333"
else:
    text_color = "#f0f0f0"
    bg_overlay_color = "rgba(20, 20, 20, 0.9)"
    chat_bg = "#2b2b2b"
    user_border = "#4da6ff"
    bot_border = "#81c784"
    header_color = "#ffffff"
    input_bg = "#2b2b2b"
    input_text = "#ffffff"

st.markdown(f"""
    <style>
    html, body, [class*="css"], .stMarkdown, .stText {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: {text_color} !important; 
    }}
    .stApp {{
        background-image: url("https://i.postimg.cc/vBh5LSLT/logo.webp");
        background-size: 50%;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        background-color: {bg_overlay_color}; 
        background-blend-mode: overlay;
    }}
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
    .stChatMessage {{
        background-color: {chat_bg} !important;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }}
    .stChatMessage p {{
        color: {text_color} !important;
    }}
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {{
        border-left: 5px solid {user_border};
    }}
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {{
        border-left: 5px solid {bot_border};
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    a {{ color: {user_border} !important; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    </style>
""", unsafe_allow_html=True)

# --- UI HEADER ---
st.markdown('<div class="main-header">Nairobi Securities Exchange</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Digital Assistant & Market Intelligence</div>', unsafe_allow_html=True)

# --- API CONNECTION FUNCTION ---
def query_api(user_query):
    try:
        response = requests.post(
            f"{API_URL}/ask", 
            json={"query": user_query},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Unable to connect to the NSE Engine. The backend might be offline or waking up."}
    except Exception as e:
        return {"error": f"System Error: {str(e)}"}

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
        with st.spinner("Analyzing market data..."):
            result = query_api(prompt)
            
            if "error" in result:
                st.error(result["error"])
                st.session_state.messages.append({"role": "assistant", "content": result["error"]})
            else:
                answer = result.get("answer", "No response.")
                sources = result.get("sources", [])
                
                full_response = answer
                if sources:
                    source_text = "\n\n**Sources:** \n" + "  \n".join([f"• [{s.replace('https://www.nse.co.ke', 'nse.co.ke').split('/')[-1]}]({s})" for s in sources])
                    st.markdown(full_response + source_text)
                    st.session_state.messages.append({"role": "assistant", "content": full_response + source_text})
                else:
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})