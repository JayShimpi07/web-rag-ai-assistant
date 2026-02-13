import streamlit as st

def init_memory():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def add_to_memory(user_query, ai_response):
    st.session_state.chat_history.append({
        "user": user_query,
        "ai": ai_response
    })

def get_memory():
    return st.session_state.chat_history

def reset_memory():
    st.session_state.chat_history = []
