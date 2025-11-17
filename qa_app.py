# app.py

import streamlit as st
from ml import qa_chain_

st.set_page_config(page_title="My RAG App", page_icon="🧠")

st.title("RAG Chatbot powered by Ollama + Streamlit")

query = st.text_input("Ask your question:")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = qa_chain_()

if st.button("Submit") and query:
    answer = st.session_state.qa_chain.run(query)
    st.write("### Answer:")
    st.write(answer)
