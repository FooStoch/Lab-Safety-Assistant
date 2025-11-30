import streamlit as st
import json
import base64
from lab_safety import LabSafetyAssistantV3
st.set_page_config(layout="wide")
# Initialize assistant once
if "assistant" not in st.session_state:
    st.session_state.assistant = LabSafetyAssistantV3()
# Holds UI-visible chat history (NOT internal model history)
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []
# Holds most recent parsed structured output for left panel
if "latest_structured" not in st.session_state:
    st.session_state.latest_structured = None
# Initialize clear counter for dynamic key
if "clear_counter" not in st.session_state:
    st.session_state.clear_counter = 0
# -------------------------------
# Layout: Left Summary + Main Chat
# -------------------------------
left_col, main_col = st.columns([1, 2.7], gap="large")
with left_col:
    with st.container(height=600):
        st.header("Safety Summary")
        data = st.session_state.latest_structured
        if data:
            def show_list(name, lst):
                if lst:
                    st.markdown(f"**{name}:**")
                    for item in lst:
                        st.markdown(f"- {item}")
            show_list("Hazards", data.get("hazards"))
            show_list("Required PPE", data.get("ppe_required"))
            show_list("Recommended PPE", data.get("ppe_recommended"))
            show_list("Immediate Actions", data.get("immediate_actions"))
            show_list("Safer Substitutes", data.get("safer_substitutes"))
            st.markdown("**Confidence:** " + str(data.get("confidence", "N/A")))
        else:
            st.write("Ask a lab safety question to see the summary.")
with main_col:
    with st.container(height=600):
        st.header("Lab Safety Chat")
        # Render chat messages
        for msg in st.session_state.ui_messages:
            if msg["role"] == "user":
                st.markdown(f"**🫵(You):** {msg['content']}")
            elif msg["role"] == "assistant":
                d = msg["content"]
                # Clean assistant output
                if "explain_short" in d:
                    st.markdown(f"**❗(Assistant):** {d['explain_short']}")
                if "official_response" in d:
                    st.markdown(f"**🧠(Assistant):** {d['official_response']}")
                if "citations" in d and d["citations"]:
                    st.markdown("**Sources:**")
                    for c in d["citations"]:
                        st.markdown(f"- {c}")
                st.divider()
    # Input box (outside the scrollable container, at the bottom)
    user_input = st.text_input("Enter a lab safety question:", key=f"text_value_{st.session_state.clear_counter}")
    uploaded_files = st.file_uploader("Upload images (optional)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True, key=f"uploader_{st.session_state.clear_counter}")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        send_btn = st.button("Send")
    with btn_col2:
        clear_btn = st.button("Clear History")
    if send_btn and (user_input.strip() or uploaded_files):
        query_parts = [user_input.strip()] if user_input.strip() else []
        for file in uploaded_files or []:
            base64_img = base64.b64encode(file.getvalue()).decode('utf-8')
            query_parts.append(f"image:{base64_img}")
        full_query = " ".join(query_parts)
        ui_content = user_input.strip()
        if uploaded_files:
            ui_content += f" (with {len(uploaded_files)} images)"
        # add user message to UI
        st.session_state.ui_messages.append(
            {"role": "user", "content": ui_content}
        )
        # call assistant
        reply = st.session_state.assistant.query(full_query)
        parsed = reply["parsed"]
        # store for UI
        st.session_state.ui_messages.append(
            {"role": "assistant", "content": parsed}
        )
        # update left summary
        st.session_state.latest_structured = parsed
        # Increment counter to force new input widget (clears it)
        st.session_state.clear_counter += 1
        st.rerun()
    if clear_btn:
        st.session_state.ui_messages = []
        st.session_state.latest_structured = None
        st.session_state.assistant = LabSafetyAssistantV3()
        # Reset counter for fresh input
        st.session_state.clear_counter = 0
        st.rerun()
