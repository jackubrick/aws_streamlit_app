import streamlit as st
from utils import rag


st.title("Question & Answering Bot")
st.write('Enter a Question:  ')

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

# Displays all produced messages on the screen:
for rag_message in st.session_state.rag_messages:
    with st.chat_message(rag_message["role"]):
        st.markdown(rag_message["content"])

if prompt := st.chat_input(placeholder="Enter your prompt."):

    user_message = {"role" : "user", "content" : prompt}
    st.session_state.rag_messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = rag.qa_chain.invoke(prompt)
        st.markdown(response)

    assistant_message = {"role" : "assistant", "content" : response}
    st.session_state.rag_messages.append(assistant_message)
