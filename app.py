# app.py — robust input using st.form to avoid text_input/button timing issues
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
    """
    Canonical key for message content. Use for deduplication.
    """
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
    # handle the case where you moved the intro into assistant.chat_history[0]
    if getattr(assistant, "chat_history", None) and len(assistant.chat_history) > 0:
        first = assistant.chat_history[0]
        if isinstance(first, dict):
            intro_text = first.get("content", "")
        else:
            intro_text = str(first)
    else:
        intro_text = ("Hello — I'm Lab Safety Assistant. Tell me about your planned experiment or upload a photo "
                      "(type 'image:<URL or dataURL>'). I'll identify hazards, required PPE, and high-level safety advice.")
    # store intro as a dict so message shape matches assistant responses (prevents shape mismatch duplicates)
    st.session_state["messages"] = [{"role": "assistant", "content": {"official_response": intro_text, "explain_short": ""}}]

if "last_parsed" not in st.session_state:
    st.session_state["last_parsed"] = {}

# Left sticky panel CSS (full-height, vertically centered, light red background)
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
  background: rgba(255,255,255,0.92);
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

# Layout
left_col, main_col = st.columns([1, 3])
left_container = left_col.container()

def render_left(parsed: Dict[str, Any]):
    if not parsed:
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
    # show short explain and official_response if present
    explain_short = parsed.get("explain_short", "")
    official_response = parsed.get("official_response", "")
    if explain_short:
        html += f"<hr><p><em>{explain_short}</em></p>"
    if official_response:
        html += f"<p><strong>Official summary:</strong></p><p>{official_response}</p>"
    # retrieved sources if present
    retrieved = parsed.get("retrieved_meta") or parsed.get("retrieved_sources") or []
    if retrieved:
        html += "<hr><p><strong>Retrieved sources:</strong></p><ul>"
        # if this is list of dicts (meta), show source + method + score when available
        if all(isinstance(x, dict) for x in retrieved):
            for r in retrieved:
                src = r.get("source") or r.get("filename") or str(r)
                method = r.get("method", "")
                score = r.get("score", "")
                html += f"<li class='mono'>{src} {f'({method}, {score:.3f})' if method else ''}</li>"
        else:
            for r in retrieved:
                html += f"<li class='mono'>{r}</li>"
        html += "</ul>"

    html += "</div></div>"
    left_container.markdown(html, unsafe_allow_html=True)

# initial left panel render
render_left(st.session_state.get("last_parsed", {}))

# Chat rendering area
with main_col:
    st.header("Lab Safety Chat")

    chat_container = st.container()

    def render_chat(max_messages: int = 200):
        """
        Render the chat while skipping duplicate messages that have identical (role, content_key).
        This prevents repeated intro/identical messages from showing multiple times in the UI.
        """
        chat_container.empty()
        with chat_container:
            msgs = st.session_state.get("messages", [])[-max_messages:]
            seen = set()  # track seen (role, content_key)
            for m in msgs:
                role = m.get("role", "assistant")
                content = m.get("content")
                key = (role, content_key(content))
                # skip duplicated content (first occurrence shown, later duplicates suppressed)
                if key in seen:
                    continue
                seen.add(key)

                if role == "user":
                    st.markdown(f"**You:** {content}")
                else:
                    if isinstance(content, dict):
                        # Prefer short explain then official_response for compact display
                        explain = content.get("explain_short", "")
                        official = content.get("official_response") or content.get("raw_text") or ""
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

    # Input widgets inside a form to ensure stable submission
    st.write("---")
    st.subheader("Ask the assistant or upload an image")
    with st.form(key="user_input_form"):
        col_text, col_file, col_send = st.columns([6,2,1])
        # text input stored in session state under form key "user_text"
        col_text.text_input("Type question or 'image:<URL or dataURL>'", key="user_text")
        # file uploader inside form (key "user_file")
        col_file.file_uploader("Upload image (optional)", type=["png","jpg","jpeg","bmp","tiff"], key="user_file")
        submitted = col_send.form_submit_button("Send")

    # If the form was submitted, handle the input (read values from session_state)
    if submitted:
        user_input = st.session_state.get("user_text", "").strip()
        uploaded_file = st.session_state.get("user_file", None)
        # If user uploaded an image but no text, auto create data url
        if uploaded_file is not None and not user_input:
            data_url = file_to_data_url(uploaded_file)
            user_input = f"image:{data_url}"

        if user_input:
            # robust append that avoids duplicates anywhere in history
            def append_message_safe(role: str, content):
                new_key = (role, content_key(content))
                # If exactly same role+content already exists anywhere, skip
                for m in st.session_state["messages"]:
                    if (m.get("role"), content_key(m.get("content"))) == new_key:
                        return
                st.session_state["messages"].append({"role": role, "content": content})

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
                    if isinstance(parsed, dict):
                        # store parsed + retrieved in session state for left panel
                        st.session_state["last_parsed"] = parsed
                        # attach retrieved meta into parsed for left panel convenience (if not already)
                        if "retrieved_meta" not in parsed and retrieved:
                            parsed["retrieved_meta"] = retrieved
                        render_left(parsed)

                        official = parsed.get("official_response") or parsed.get("raw_text") or ""
                        assistant_display = {"official_response": official, **{k:v for k,v in parsed.items() if k not in ('official_response',)}}
                        append_message_safe("assistant", assistant_display)
                    else:
                        append_message_safe("assistant", str(parsed))
                    render_chat()

    # Clear chat (keeps left panel)
    if st.button("Clear chat"):
        # reset messages to initial intro only (read from assistant.chat_history[0] if present)
        if getattr(assistant, "chat_history", None) and len(assistant.chat_history) > 0:
            first = assistant.chat_history[0]
            if isinstance(first, dict):
                intro = first.get("content", "")
            else:
                intro = str(first)
        else:
            intro = ("Hello — I'm Lab Safety Assistant. Tell me about your planned experiment or upload a photo "
                     "(type 'image:<URL or dataURL>'). I'll identify hazards, required PPE, and high-level safety advice.")
        # store as dict shape to match assistant responses
        st.session_state["messages"] = [{"role": "assistant", "content": {"official_response": intro, "explain_short": ""}}]
        st.session_state["last_parsed"] = {}
        render_left({})
        render_chat()
