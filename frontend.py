import os
import requests
import streamlit as st

# ----------------------------
# Config
# ----------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")  # Flask backend
st.set_page_config(page_title="🎬 Movie Chat", layout="centered")

st.title("🎬 Movie Chat")
st.caption("Ask about movies naturally. Follow up with more questions — the chat keeps context.")

# ----------------------------
# Session state
# ----------------------------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}

# ----------------------------
# Sidebar: controls
# ----------------------------
with st.sidebar:
    st.markdown("**Backend:**")
    st.text(API_URL)
    if st.button("🧹 New chat"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.rerun()

# ----------------------------
# Render history
# ----------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ----------------------------
# Chat input
# ----------------------------
prompt = st.chat_input("Type your question (e.g., “top 5 action movies after 2015”)")
if prompt:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call backend /chat
    try:
        payload = {"message": prompt}
        if st.session_state.conversation_id:
            payload["conversation_id"] = st.session_state.conversation_id

        res = requests.post(f"{API_URL}/chat", json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()

        # Save/update conversation id
        st.session_state.conversation_id = data.get("conversation_id", st.session_state.conversation_id)

        # Assistant reply
        assistant_msg = data.get("assistant_message", "Sorry, I didn't get that.")
        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})

        with st.chat_message("assistant"):
            st.write(assistant_msg)

    except Exception as e:
        err = f"Request failed: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)
