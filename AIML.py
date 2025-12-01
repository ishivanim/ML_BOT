import streamlit as st
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import joblib

from ml import qa_chain_ #Importing the built qa_chain from ml.py file

emd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_ = joblib.load("embedding.pkl")

ml_vectore_store = Chroma(
    collection_name="my_ml_vector_data",
    persist_directory="./db",
    embedding_function=emd
)

dl_vectore_store = Chroma(
    collection_name="my_dl_vector_data",
    persist_directory="./db",
    embedding_function=emd
)

ai_vectore_store = Chroma(
    collection_name="my_ai_vector_data",
    persist_directory="./db",
    embedding_function=emd
)

ml_retriever = ml_vectore_store.as_retriever(search_kwargs={"k": 3})
dl_retriever = dl_vectore_store.as_retriever(search_kwargs={"k": 3})
ai_retriever = ai_vectore_store.as_retriever(search_kwargs={"k": 3})

st.set_page_config(page_title="My RAG App", page_icon="🧠")

st.title("AIML-BOT")

query = st.text_input("Ask your question:")

query_ = emd.embed_query(query)
query_ = np.array(query_)

def cosine_similarity(question, emb):
    similarity = np.dot(question,emb)/(np.linalg.norm(question)*np.linalg.norm(emb))
    return similarity

ml_similarity = cosine_similarity(query_, np.array(embedding_["ml"]).squeeze())
dl_similarity = cosine_similarity(query_, np.array(embedding_["dl"]).squeeze())
ai_similarity = cosine_similarity(query_, np.array(embedding_["ai"]).squeeze())

if ml_similarity>dl_similarity and ml_similarity>ai_similarity:
    retriever = ml_retriever
elif dl_similarity>ml_similarity and dl_similarity>ai_similarity:
    retriever = dl_retriever
elif ai_similarity>ml_similarity and ai_similarity>dl_similarity:
    retriever = ai_retriever

if st.button("Submit") and query:
    chain = qa_chain_(retriever)
    answer = chain.run(query)
    st.write("### Answer:")
    st.write(answer)
