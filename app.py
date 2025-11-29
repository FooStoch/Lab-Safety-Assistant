# === app.py (edited) ===
# Key edits:
#  - Fix intro seeding bug (intro_text is a string on assistant)
#  - Ensure left panel is cleared before re-rendering and always vertically centered (uses fixed positioning)
#  - Store last_retrieved in session_state and show explain_short + official_response + retrieved sources in the left panel
#  - Improve render_chat to show assistant's official_response/explain_short cleanly and avoid repeated appended UI elements

import os
import base64
import mimetypes
import json
import streamlit as st
from typing import Optional, Dict, Any

import lab_safety  # your module

# Page config
st.set_page_config(page_title="Lab Safety Assistant", layout="wide")

# Secrets and API key override
if "OPENROUTER_API_KEY" not in st.secrets:
    st.warning("OPENROUTER_API_KEY not found in Streamlit secrets.")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", None)
if OPENROUTER_KEY:
    lab_safety.OPENROUTER_API_KEY = OPENROUTER_KEY

# Assume output_txt is set as OUTPUT_DIR in lab_safety.py (./output_txt)
DOCS_DIR = getattr(lab_safety, "OUTPUT_DIR", "./output_txt")

# Helpers
def file_to_data_url(uploaded) -> Optional[str]:
    if uploaded is None:
        return None
    data = uploaded.read()
    mime, _ = mimetypes.guess_type(uploaded.name)
    if not mime:
        mime = "application/octet-stream"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def content_key(obj):
    try:
        if isinstance(obj, str):
            return obj.strip()
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return str(obj)

# Initialize assistant once (cached resource to avoid reloading heavy TF-IDF)
@st.cache_resource(show_spinner=False)
def init_assistant(docs_dir: str):
    return lab_safety.LabSafetyAssistantV3(docs_dir=docs_dir)

if not os.path.isdir(DOCS_DIR):
    st.sidebar.error(f"Documents folder not found at {DOCS_DIR}. Ensure output_txt/ is in repo root or update lab_safety.OUTPUT_DIR")

with st.spinner("Loading documents and building vectorizers..."):
    assistant = init_assistant(DOCS_DIR)

# Session state initialization
if "messages" not in st.session_state:
    # seed with assistant intro (only once)
    intro = assistant.intro_text if hasattr(assistant, 'intro_text') else (
        "Hello — I'm Lab Safety Assistant. Tell me about your planned experiment or upload a photo "
        "(type 'image:<URL or dataURL>'). I'll identify hazards, required PPE, and high-level safety advice.")
    st.session_state["messages"] = [{"role": "assistant", "content": intro}]

if "last_parsed" not in st.session_state:
    st.session_state["last_parsed"] = {}

if "last_retrieved" not in st.session_state:
    st.session_state["last_retrieved"] = []

# Left sticky panel CSS (full-height, vertically centered, light red background).
# Use fixed positioning within the left column area to keep contents centered while scrolling.
LEFT_PANEL_CSS = """
<style>
.left-sticky {
  position: fixed;
  left: 8px; /* small inset so it aligns within Streamlit column */
  top: 0;
  bottom: 0;
  height: 100vh;
  display: flex;
  align-items: center; /* vertical centering */
  justify-content: center;
  background: #FFD6D7;
  padding: 18px;
  box-sizing: border-box;
  z-index: 10;
}
.left-box {
  width: 100%;
  border-radius: 8px;
  background: rgba(255,255,255,0.96);
  padding: 14px 16px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06);
  max-height: 92vh;
  overflow-y: auto;
  overflow-wrap: anywhere;
}
.left-box h3 { margin-bottom: 6px; margin-top: 0; }
.left-box p { margin: 6px 0; }
.left-box .mono { font-family: monospace; font-size: 0.95em; }
</style>
"""
st.markdown(LEFT_PANEL_CSS, unsafe_allow_html=True)

# Layout
left_col, main_col = st.columns([1, 3])
# Create a persistent left container (we will .empty() it before rendering to replace contents)
left_container = left_col.container()


def render_left(parsed: Dict[str, Any], retrieved: Optional[list] = None):
    # replace the left container contents (prevent stacking duplicates)
    left_container.empty()

    # Compose HTML content
    if not parsed or (isinstance(parsed, dict) and not any(parsed.get(k) for k in ("hazards","ppe_required","official_response","explain_short","retrieved_sources"))):
        html = "<div class='left-sticky'><div class='left-box'><h3>Safety Summary</h3><p>No model output yet.</p></div></div>"
        left_container.markdown(html, unsafe_allow_html=True)
        return

    hazards = parsed.get("hazards", [])
    ppe_required = parsed.get("ppe_required", [])
    ppe_recommended = parsed.get("ppe_recommended", [])
    immediate_actions = parsed.get("immediate_actions", [])
    safer_substitutes = parsed.get("safer_substitutes", [])
    citations = parsed.get("citations", [])
    confidence = parsed.get("confidence", "")
    explain_short = parsed.get("explain_short", "")
    official_response = parsed.get("official_response", "")
    retrieved_sources = parsed.get("retrieved_sources", []) if parsed else []

    def rows(title, items):
        if not items:
            return f"<p><strong>{title}:</strong> —</p>"
        if isinstance(items, str):
            items_str = items
        else:
            # if list of dicts (retrieved), stringify nicely
            if all(isinstance(x, dict) and 'source' in x for x in items):
                items_str = ", ".join([f"{x.get('source')} ({x.get('method')}, score={x.get('score'):.2f})" for x in items])
            else:
                items_str = ", ".join([str(x) for x in items])
        return f"<p><strong>{title}:</strong> {items_str}</p>"

    html = "<div class='left-sticky'><div class='left-box'>"
    html += "<h3>Safety Summary</h3>"
    if explain_short:
        html += f"<p><em>{explain_short}</em></p>"
    html += rows("Hazards", hazards)
    html += rows("PPE (required)", ppe_required)
    html += rows("PPE (recommended)", ppe_recommended)
    html += rows("Immediate actions", immediate_actions)
    html += rows("Safer substitutes", safer_substitutes)
    html += rows("Citations", citations)
    html += rows("Confidence", confidence)
    # show official_response and retrieved sources below
    if official_response:
        html += f"<hr><p><strong>Official summary:</strong></p><p>{official_response}</p>"
    if retrieved is not None:
        # prefer full meta when available
        html += "<hr><p><strong>Retrieved sources (top matches):</strong></p>"
        if retrieved:
            html += "<ul>"
            for r in retrieved:
                html += f"<li class='mono'>{r.get('source')} — method={r.get('method')}, score={r.get('score'):.3f}</li>"
            html += "</ul>"
        else:
            html += "<p>— none</p>"

    html += "</div></div>"
    left_container.markdown(html, unsafe_allow_html=True)

# initial left panel render
render_left(st.session_state.get("last_parsed", {}), st.session_state.get("last_retrieved", []))

# Chat rendering area
with main_col:
    st.header("Lab Safety Chat")

    chat_container = st.container()

    def render_chat(max_messages: int = 50):
        # Clear and re-render last up-to max_messages
        chat_container.empty()
        with chat_container.container():
            # display only the most recent `max_messages` entries to avoid overwhelming UI
            msgs = st.session_state.get("messages", [])[-max_messages:]
            for m in msgs:
                role = m.get("role", "assistant")
                content = m.get("content")
                if role == "user":
                    st.markdown(f"**You:** {content}")
                else:
                    # Assistant messages may be strings or structured dicts
                    if isinstance(content, dict):
                        # Prefer official_response and explain_short for compact display
                        official = content.get("official_response") or content.get("raw_text") or ""
                        explain = content.get("explain_short", "")
                        if explain:
                            st.markdown(f"**Assistant:** _{explain}_")
                        if official:
                            st.markdown(f"**Assistant (summary):** {official}")
                        with st.expander("Full structured output (click to expand)"):
                            st.json(content)
                    else:
                        st.markdown(f"**Assistant:** {content}")

    # Render chat once at load
    render_chat()

    # Input widgets
    st.write("---")
    st.subheader("Ask the assistant or upload an image")
    col_text, col_file, col_send = st.columns([6,2,1])
    user_input = col_text.text_input("Type question or 'image:<URL or dataURL>'", key="user_input")
    uploaded_file = col_file.file_uploader("Upload image (optional)", type=["png","jpg","jpeg","bmp","tiff"])
    send_btn = col_send.button("Send")

    # If user uploaded an image but no text, auto create data url
    if uploaded_file is not None and not user_input:
        data_url = file_to_data_url(uploaded_file)
        user_input = f"image:{data_url}"

    def append_message_safe(role: str, content):
        new_key = (role, content_key(content))
        for m in st.session_state["messages"]:
            if (m.get("role"), content_key(m.get("content"))) == new_key:
                return
        st.session_state["messages"].append({"role": role, "content": content})

    # On send
    if send_btn and user_input:
        append_message_safe("user", user_input)
        render_chat()

        with st.spinner("Querying model (this may take a few seconds)..."):
            try:
                result = assistant.query(user_input)
            except Exception as e:
                err = f"Model call failed: {e}"
                append_message_safe("assistant", err)
                render_chat()
            else:
                parsed = result.get("parsed", {})
                retrieved = result.get("retrieved", [])
                # store parsed + retrieved in session state for left panel
                if isinstance(parsed, dict):
                    st.session_state["last_parsed"] = parsed
                    st.session_state["last_retrieved"] = retrieved
                    # update left panel with both parsed and retrieved meta
                    render_left(parsed, retrieved)

                    # Prepare assistant message: compact keys first
                    official = parsed.get("official_response") or parsed.get("raw_text") or ""
                    explain = parsed.get("explain_short", "")
                    # Build a deterministic dict for storage/display
                    assistant_display = {
                        "explain_short": explain,
                        "official_response": official,
                        # keep the rest so the expander shows full structure
                        **{k:v for k,v in parsed.items() if k not in ("explain_short","official_response")} 
                    }
                    append_message_safe("assistant", assistant_display)
                else:
                    append_message_safe("assistant", str(parsed))
                render_chat()

    # Clear chat (keeps left panel)
    if st.button("Clear chat"):
        intro = assistant.intro_text if hasattr(assistant, 'intro_text') else ("Hello — I'm Lab Safety Assistant.")
        st.session_state["messages"] = [{"role": "assistant", "content": intro}]
        st.session_state["last_parsed"] = {}
        st.session_state["last_retrieved"] = []
        render_left({}, [])
        render_chat()
