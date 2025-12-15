"""
Companion AI Chat Interface
A Streamlit app that provides a child-safe AI companion chat interface.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
from chat_backend import process_message, memory_manager

# -----------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Companion AI",
    page_icon="🤖",
    layout="wide",
)

# -------------------------
# Session State Initialization
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "freeze_input" not in st.session_state:
    st.session_state.freeze_input = False
if "pending" not in st.session_state:
    st.session_state.pending = False
if "input_box" not in st.session_state:
    st.session_state.input_box = ""


# -------------------------
# UI Styling
# -------------------------
CUSTOM_CSS = """
<style>

    body {
        color: black !important;
    }

    .user-bubble {
        background-color: #e8f0fe;
        padding: 10px 15px;
        border-radius: 12px;
        margin-bottom: 6px;
        max-width: 80%;
        color: black;
    }

    .bot-bubble {
        background-color: #f1f3f4;
        padding: 10px 15px;
        border-radius: 12px;
        margin-bottom: 6px;
        max-width: 80%;
        color: black;
    }

    .user-wrapper {
        display: flex;
        justify-content: flex-end;
    }

    .debug-box {
        background-color: #fafafa;
        padding: 10px 15px;
        border-left: 4px solid #4285f4;
        margin-top: 8px;
        color: #333;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------
# Chat History Display
# -------------------------
def render_history():
    """Render the chat history from session state."""
    for role, text, explanation in st.session_state.history:
        if role == "user":
            st.markdown(
                f"""
                <div class="user-wrapper">
                    <div class="user-bubble">{st.markdown(escape_html(text), unsafe_allow_html=True) if False else text}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                f"""
                <div class="bot-bubble">{text}</div>
            """,
                unsafe_allow_html=True,
            )

            if explanation:
                with st.expander("Why did the model say this?"):
                    st.markdown(
                        f"""
                        <div class="debug-box">
                        <b>Retrieved Memory:</b><br>{explanation.get("retrieved", [])}<br><br>
                        <b>Entities:</b> {explanation.get("entities", [])}<br><br>
                        <b>Timeline Summary:</b><br>{explanation.get("timeline", "")}<br><br>
                        <b>Detected Emotion:</b> {explanation.get("emotion")}<br>
                        <b>Safety:</b> {explanation.get("safety")}<br>
                        <b>Behavior:</b> {explanation.get("bot_behavior")}<br>
                        <b>Error (if any):</b> {explanation.get("error", "None")}<br>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------
def escape_html(s):
    """Escape HTML special characters in a string."""
    return (s or "").replace("<", "&lt;").replace(">", "&gt;")


# -------------------------
# Chat Input
# -------------------------
def user_input_box():
    """Render the user input box and return the entered text."""
    return st.text_input(
        "Type your message:",
        value=st.session_state.input_box,
        key="input_box_widget",
        placeholder="Say something...",
        disabled=st.session_state.freeze_input or st.session_state.pending,
    )


# -------------------------
# Main Chat Logic
# -------------------------
st.title("Companion AI 🤖")

render_history()

col_input, col_btn = st.columns([8, 1])
with col_input:
    user_text = user_input_box()
with col_btn:
    send_btn = st.button("Send", disabled=st.session_state.pending)

if st.button("Clear chat"):
    st.session_state.history = []
    try:
        with st.spinner("Clearing Chats..."):
            memory_manager.reset()
    except Exception:
        pass
    st.session_state.input_box = ""
    st.session_state.freeze_input = False
    st.session_state.pending = False
    st.rerun()


# -----------------------------------------------------
# Handle User Message Submission
# -----------------------------------------------------
def handle_user_message():
    """Handle user message submission."""
    if st.session_state.pending:
        return

    text = (user_text or "").strip()
    if not text:
        return

    st.session_state.pending = True
    st.session_state.freeze_input = True

    st.session_state.history.append(("user", text, None))

    st.session_state.history.append(("assistant", "Thinking...", None))

    st.rerun()


# -----------------------------------------------------
# Handle Send Button Click
# -----------------------------------------------------
if send_btn:
    if not st.session_state.pending:
        handle_user_message()
    else:
        pass

# -----------------------------------------------------
# Process Pending Assistant Response
# -----------------------------------------------------
if st.session_state.pending:
    if st.session_state.history and st.session_state.history[-1][1] == "Thinking...":
        if (
            len(st.session_state.history) >= 2
            and st.session_state.history[-2][0] == "user"
        ):
            local_user_text = st.session_state.history[-2][1]
        else:
            local_user_text = ""

        with st.spinner("Companion is thinking..."):
            try:
                reply, explanation = process_message(local_user_text, debug=True)
            except Exception as e:
                reply = "I'm sorry — something went wrong while generating a response."
                explanation = {
                    "retrieved": [],
                    "entities": [],
                    "timeline": "",
                    "emotion": "unknown",
                    "safety": "error",
                    "bot_behavior": "error",
                    "error": str(e),
                }

        if (
            st.session_state.history
            and st.session_state.history[-1][1] == "Thinking..."
        ):
            st.session_state.history[-1] = ("assistant", reply, explanation)
        else:
            st.session_state.history.append(("assistant", reply, explanation))

        st.session_state.input_box = ""
        st.session_state.freeze_input = False
        st.session_state.pending = False

        st.rerun()

st.markdown("---")
st.caption("Child-safe empathetic assistant — powered by Mistral + LoRA")
