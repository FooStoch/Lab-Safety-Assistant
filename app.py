# app.py
import os
import base64
import mimetypes
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any

# Put your lab_safety.py in the same folder. We will import it and then override the API key from secrets.
import lab_safety

# ---- Page config ----
st.set_page_config(page_title="Lab Safety Assistant", layout="wide")

# ---- Secrets / config ----
if "OPENROUTER_API_KEY" not in st.secrets:
    st.warning("OpenRouter API key not found in Streamlit secrets. Add OPENROUTER_API_KEY to secrets.toml.")
OPENROUTER_KEY = st.secrets.get("OPENROUTER_API_KEY", None)
if OPENROUTER_KEY:
    # override the module-level constant in lab_safety so the code uses the secret
    lab_safety.OPENROUTER_API_KEY = OPENROUTER_KEY

# docs dir: default to local ./output_txt, allow override with secrets DOCS_DIR
DEFAULT_DOCS = "./output_txt"
DOCS_DIR = st.secrets.get("DOCS_DIR", DEFAULT_DOCS)

# ---- Helper: convert uploaded file to data URL ----
def file_to_data_url(uploaded) -> Optional[str]:
    """
    uploaded is a Streamlit UploadedFile object -- convert to a data:... base64 url
    """
    if uploaded is None:
        return None
    data = uploaded.read()
    # detect mime type if possible
    mime, _ = mimetypes.guess_type(uploaded.name)
    if not mime:
        mime = "application/octet-stream"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ---- Initialize assistant (cache to avoid reloading repeatedly) ----
@st.cache_resource(show_spinner=False)
def init_assistant(docs_dir: str):
    # fix lab_safety's output dir if it's set wrong in the file
    # (lab_safety.LabSafetyAssistantV3 accepts docs_dir argument)
    return lab_safety.LabSafetyAssistantV3(docs_dir=docs_dir)

# If docs folder does not exist, show message and let user pick a folder upload route
if not os.path.isdir(DOCS_DIR):
    st.sidebar.error(
        f"Documents folder not found at {DOCS_DIR}. Please (1) put 'output_txt' in the repo root or (2) set DOCS_DIR in Streamlit secrets."
    )

# instantiate assistant (this builds vectorizers and may take a moment)
with st.spinner("Loading SDS documents and vectorizers..."):
    assistant = init_assistant(DOCS_DIR)

# ---- Session state: store chat messages and last parsed structured info ----
if "messages" not in st.session_state:
    # initialize with assistant intro from the assistant instance
    intro = assistant.chat_history[0]["content"] if assistant.chat_history else "Hello — Lab Safety Assistant."
    st.session_state["messages"] = [{"role": "assistant", "content": intro}]
if "last_parsed" not in st.session_state:
    st.session_state["last_parsed"] = {}

# ---- Layout: left fixed panel (structured safety fields) + main chat column ----
left_col, main_col = st.columns([1, 3])

# left panel styling: vertically center content via a small markdown block with CSS
left_panel_container = left_col.empty()

LEFT_PANEL_CSS = """
<style>
.left-panel {
  height: 80vh;
  display: flex;
  align-items: center;
  justify-content: center;
}
.left-box {
  width: 100%;
  padding: 18px;
  border-radius: 8px;
  background: #f7f9fc;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  overflow-wrap: anywhere;
  max-height: 76vh;
}
.left-box h3 { margin-bottom: 6px; }
.left-box p { margin: 6px 0; }
</style>
"""
st.markdown(LEFT_PANEL_CSS, unsafe_allow_html=True)

def render_left(parsed: Dict[str, Any]):
    # parsed may be raw text or dict
    if not parsed:
        content = "<div class='left-panel'><div class='left-box'><h3>Safety Summary</h3><p>No model output yet.</p></div></div>"
        left_panel_container.markdown(content, unsafe_allow_html=True)
        return
    # if the parsed is nested (model returned raw_text) handle gracefully
    if isinstance(parsed, dict):
        hazards = parsed.get("hazards", [])
        ppe_required = parsed.get("ppe_required", [])
        ppe_recommended = parsed.get("ppe_recommended", [])
        immediate_actions = parsed.get("immediate_actions", [])
        safer_substitutes = parsed.get("safer_substitutes", [])
        citations = parsed.get("citations", [])
        confidence = parsed.get("confidence", "")
    else:
        # fallback: show raw text
        hazards = ppe_required = ppe_recommended = immediate_actions = safer_substitutes = citations = []
        confidence = ""
    # build HTML
    html = "<div class='left-panel'><div class='left-box'>"
    html += "<h3>Safety Summary</h3>"
    def rows(title, items):
        if not items:
            return f"<p><strong>{title}:</strong> —</p>"
        if isinstance(items, str):
            items_str = items
        else:
            items_str = ", ".join([str(x) for x in items])
        return f"<p><strong>{title}:</strong> {items_str}</p>"
    html += rows("Hazards", hazards)
    html += rows("PPE (required)", ppe_required)
    html += rows("PPE (recommended)", ppe_recommended)
    html += rows("Immediate actions", immediate_actions)
    html += rows("Safer substitutes", safer_substitutes)
    html += rows("Citations", citations)
    html += rows("Confidence", confidence)
    html += "</div></div>"
    left_panel_container.markdown(html, unsafe_allow_html=True)

# initial render
render_left(st.session_state.get("last_parsed", {}))

# ---- Main chat UI ----
with main_col:
    st.header("Lab Safety Chat")

    # Chat history container (scrollable)
    chat_container = st.container()
    with st.expander("Conversation (expand/collapse)", expanded=False):
        st.write("This shows the full assistant chat history (also displayed below).")

    # display messages (user & assistant) in order
    def render_chat():
        chat_container.empty()
        with chat_container:
            for m in st.session_state["messages"]:
                role = m.get("role", "assistant")
                content = m.get("content")
                if role == "user":
                    st.markdown(f"**You:** {content}")
                else:
                    # assistant may provide structured dictionary or plain string
                    if isinstance(content, dict):
                        # show official_response (if present) then extra fields in expander
                        official = content.get("official_response") or content.get("raw_text") or ""
                        st.markdown(f"**Assistant:** {official}")
                        with st.expander("Full structured output (click to expand)"):
                            st.json(content)
                    else:
                        st.markdown(f"**Assistant:** {content}")

    render_chat()

    # ---- Input area: text, image upload, submit ----
    st.write("---")
    st.subheader("Ask the assistant or upload an image")

    col_text, col_file, col_send = st.columns([6,2,1])
    user_input = col_text.text_input("Type question or 'image:<URL or dataURL>'", key="user_input")
    uploaded_file = col_file.file_uploader("Upload image (optional)", type=["png","jpg","jpeg","bmp","tiff"])
    send_btn = col_send.button("Send")

    # Shortcut: if user uploaded an image and didn't type anything, create an image:... input
    if uploaded_file is not None and not user_input:
        data_url = file_to_data_url(uploaded_file)
        user_input = f"image:{data_url}"

    # When send pressed:
    if send_btn and user_input:
        # append user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        render_chat()

        # call assistant (show spinner)
        with st.spinner("Querying model (this may take a few seconds)..."):
            try:
                result = assistant.query(user_input)
            except Exception as e:
                st.error(f"Model call failed: {e}")
                # append error assistant message
                st.session_state["messages"].append({"role": "assistant", "content": f"Error: {e}"})
                render_chat()
            else:
                parsed = result.get("parsed", {})
                # ensure parsed is JSON/dict if possible
                if isinstance(parsed, dict):
                    # set last parsed for left panel
                    st.session_state["last_parsed"] = parsed
                    render_left(parsed)
                    # For chat display, show official_response (or raw model string)
                    official = parsed.get("official_response") or parsed.get("raw_text") or ""
                    assistant_display = {"official_response": official, **{k:v for k,v in parsed.items() if k not in ('official_response',)}}
                    st.session_state["messages"].append({"role": "assistant", "content": assistant_display})
                else:
                    # if parsed is just text, show it
                    st.session_state["messages"].append({"role": "assistant", "content": str(parsed)})
                render_chat()

    # allow clearing the whole chat (keeps left panel content as is)
    if st.button("Clear chat"):
        st.session_state["messages"] = [{"role": "assistant", "content": assistant.chat_history[0]["content"] if assistant.chat_history else "Hello"}]
        st.session_state["last_parsed"] = {}
        render_left({})
        render_chat()
