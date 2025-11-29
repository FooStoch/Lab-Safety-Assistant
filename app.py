import streamlit as st
from lab_safety import LabSafetyAssistantV3

# ---------------------------------------------
# Initialize session state
# ---------------------------------------------
if "assistant" not in st.session_state:
    st.session_state.assistant = LabSafetyAssistant()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------------------------
# Page setup
# ---------------------------------------------
st.set_page_config(page_title="Lab Safety Assistant", layout="wide")

left_col, main_col = st.columns([1, 2])

# ---------------------------------------------
# Render Safety Summary (left column)
# ---------------------------------------------
with left_col:
    st.header("Safety Summary")
    last_msg = st.session_state["messages"][-1] if st.session_state["messages"] else None

    if last_msg and isinstance(last_msg.get("content"), dict):
        st.write(last_msg["content"].get("official_response", ""))
    else:
        st.write("No model output yet.")

# ---------------------------------------------
# Main Chat UI (right column)
# ---------------------------------------------
with main_col:
    st.header("Lab Safety Chat")

    # ----- FIX: persistent container only -----
    chat_container = st.container()

    # Render the chat exactly once
    with chat_container:
        for m in st.session_state["messages"]:
            role = m.get("role", "assistant")
            content = m.get("content")

            if role == "user":
                st.markdown(f"**You:** {content}")

            else:  # assistant
                if isinstance(content, dict):
                    summary = (
                        content.get("official_response") or
                        content.get("raw_text") or
                        ""
                    )
                    st.markdown(f"**Assistant (summary):** {summary}")
                    with st.expander("Full structured output (click to expand)"):
                        st.json(content)
                else:
                    st.markdown(f"**Assistant:** {content}")

    # ----- User input form -----
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Message:")
        submitted = st.form_submit_button("Send")

        if submitted and user_input.strip():
            # Add user message
            st.session_state["messages"].append({
                "role": "user",
                "content": user_input.strip()
            })

            # Run agent
            reply = st.session_state.assistant.process(user_input.strip())

            # Add assistant reply
            st.session_state["messages"].append({
                "role": "assistant",
                "content": reply
            })

            st.rerun()
