# app.py (updated)
import os
import base64
import mimetypes
import json
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any

# import your assistant module
import lab_safety

# ---- Page config ----
st.set_page_config(page_title="Lab Safety Assistant", layout="wide")

# ---- Secrets / config ----
if "OPENROUTER_API_KEY" not in st.secrets:
    st.warning("OpenRouter API key not found in Streamlit secrets. Add OPENROUTER_API_KEY to secrets.toml.")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", None)
if OPENROUTER_KEY:
    lab_safety.OPENROUTER_API_KEY = OPENROUTER_KEY

# default docs dir
DEFAULT_DOCS = "./output_txt"
DOCS_DIR = DEFAULT_DOCS  # we assume you set OUTPUT_DIR in lab_safety.py to "./output_txt"

# ---- Helpers ----
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
    """Create a canonical key for message content for simple equality checks."""
    try:
        if isinstance(obj, str):
            return obj
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return str(obj)

# ---- Initialize assistant (cached) ----
@st.cache_resource(show_spinner=False)
def init_assistant(docs_dir: str):
    return lab_safety.LabSafetyAssistantV3(docs_dir=docs_dir)

if not os.path.isdir(DOCS_DIR):
    st.sidebar.error(f"Documents folder not found at {DOCS_DIR}. Put output_txt/ in the repo root or update lab_safety.OUTPUT_DIR")

with st.spinner("Loading SDS documents and vectorizers..."):
    assistant = init_assistant(DOCS_DIR)

# ---- Session state init ----
if "messages" not in st.session_state:
    # initialize messages with assistant intro (only once)
    intro = assistant.chat_history[0]["content"] if assistant.chat_history else "Hello — Lab Safety Assistant."
    st.session_state["messages"] = [{"role": "assistant", "content": intro}]

if "last_parsed" not in st.session_state:
    st.session_state["last_parsed"] = {}

# ---- Page layout: left sticky panel + main chat column ----
left_col, main_col = st.columns([1, 3])

# CSS for left sticky panel (full height, vertically centered, light red background)
LEFT_PANEL_CSS = """
<style>
.left-sticky {
  position: -webkit-sticky;
  position: sticky;
  top: 0;
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #FFD6D7;
  padding: 18px;
  box-sizing: border-box;
}
.left-box {
  width: 100%;
  border-radius: 8px;
  background: rgba(255,255,255,0.86);
  padding: 12px 14px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06);
  max-height: 92vh;
  overflow-wrap: anywhere;
}
.left-box h3 { margin-bottom: 6px; margin-top: 0; }
.left-box p { margin: 6px 0; }
</style>
"""
st.markdown(LEFT_PANEL_CSS, unsafe_allow_html=True)

left_container = left_col.container()

def render_left(parsed: Dict[str, Any]):
    # build the left panel content and render inside left_container
    if not parsed:
        html = "<div class='left-sticky'><div class='left-box'><h3>Safety Summary</h3><p>No model output yet.</p></div></div>"
        left_container.markdown(html, unsafe_allow_html=True)
        return

    # unpack fields safely
    hazards = parsed.get("hazards", [])
    ppe_required = parsed.get("ppe_required", [])
    ppe_recommended = parsed.get("ppe_recommended", [])
    immediate_actions = parsed.get("immediate_actions", [])
    safer_substitutes = parsed.get("safer_substitutes", [])
    citations = parsed.get("citations", [])
    confidence = parsed.get("confidence", "")

    def rows(title, items):
        if not items:
            return f"<p><strong>{title}:</strong> —</p>"
        if isinstance(items, str):
            items_str = items
        else:
            items_str = ", ".join([str(x) for x in items])
        return f"<p><strong>{title}:</strong> {items_str}</p>"

    html = "<div class='left-sticky'><div class='left-box'>"
    html += "<h3>Safety Summary</h3>"
    html += rows("Hazards", hazards)
    html += rows("PPE (required)", ppe_required)
    html += rows("PPE (recommended)", ppe_recommended)
    html += rows("Immediate actions", immediate_actions)
    html += rows("Safer substitutes", safer_substitutes)
    html += rows("Citations", citations)
    html += rows("Confidence", confidence)
    html += "</div></div>"
    left_container.markdown(html, unsafe_allow_html=True)

# initial left panel
render_left(st.session_state.get("last_parsed", {}))

# ---- Main chat UI (no expanders; full accumulator chat) ----
with main_col:
    st.header("Lab Safety Chat")

    chat_container = st.container()

    def render_chat():
        """Render the full chat history (called when changes occur)."""
        chat_container.empty()
        with chat_container:
            for m in st.session_state["messages"]:
                role = m.get("role", "assistant")
                content = m.get("content")
                if role == "user":
                    st.markdown(f"**You:** {content}")
                else:
                    if isinstance(content, dict):
                        official = content.get("official_response") or content.get("raw_text") or ""
                        st.markdown(f"**Assistant:** {official}")
                        with st.expander("Full structured output (click to expand)"):
                            st.json(content)
                    else:
                        st.markdown(f"**Assistant:** {content}")

    # render chat once
    render_chat()

    # ---- Input area (text + file upload) ----
    st.write("---")
    st.subheader("Ask the assistant or upload an image")
    col_text, col_file, col_send = st.columns([6,2,1])
    user_input = col_text.text_input("Type question or 'image:<URL or dataURL>'", key="user_input")
    uploaded_file = col_file.file_uploader("Upload image (optional)", type=["png","jpg","jpeg","bmp","tiff"])
    send_btn = col_send.button("Send")

    # If user uploaded a file but didn't type, auto-create an image:data URL
    if uploaded_file is not None and not user_input:
        data_url = file_to_data_url(uploaded_file)
        user_input = f"image:{data_url}"

    # helper: safe append to session_state messages without duplicates
    def append_message_safe(role: str, content):
        new_key = content_key(content)
        last = st.session_state["messages"][-1] if st.session_state["messages"] else None
        last_key = content_key(last["content"]) if last is not None else None
        last_role = last["role"] if last is not None else None
        # avoid consecutive identical messages of same role
        if last and last_role == role and last_key == new_key:
            return
        st.session_state["messages"].append({"role": role, "content": content})

    # When Send pressed
    if send_btn and user_input:
        # append user message safely
        append_message_safe("user", user_input)
        render_chat()

        # call assistant
        with st.spinner("Querying model (this may take a few seconds)..."):
            try:
                result = assistant.query(user_input)
            except Exception as e:
                err_text = f"Model call failed: {e}"
                append_message_safe("assistant", err_text)
                render_chat()
            else:
                parsed = result.get("parsed", {})
                # update left panel if parsed is dict
                if isinstance(parsed, dict):
                    st.session_state["last_parsed"] = parsed
                    render_left(parsed)

                    official = parsed.get("official_response") or parsed.get("raw_text") or ""
                    assistant_display = {"official_response": official, **{k:v for k,v in parsed.items() if k not in ('official_response',)}}
                    append_message_safe("assistant", assistant_display)
                else:
                    append_message_safe("assistant", str(parsed))
                render_chat()

    # clear chat button (keeps left panel content)
    if st.button("Clear chat"):
        # Reset messages to initial assistant intro only
        intro = assistant.chat_history[0]["content"] if assistant.chat_history else "Hello — Lab Safety Assistant."
        st.session_state["messages"] = [{"role": "assistant", "content": intro}]
        st.session_state["last_parsed"] = {}
        render_left({})
        render_chat()
