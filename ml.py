import regex as re
import string as sr
from langchain_text_splitters import (MarkdownTextSplitter)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Function for the preprocessing of the dataset
def preprocessing(text):
    def remove_html(text):
        pattern = re.compile(r"<,*?>")
        return pattern.sub(" ",text)
    
    def remove_punc(text):
        puncs = sr.punctuation
        return text.translate(str.maketrans(' ', ' ', puncs))
    
    def remove_url(text):
        pattern = re.compile(r"https?://\S+www\.\S+")
        return pattern.sub(' ',text)
    
    text = remove_html(text)
    text = remove_punc(text)
    text = (remove_url(text))

    return text.lower().strip()

# Load the dataset
with open('machine_learning_dataset.md', 'r', encoding="utf-8") as f:
    content = f.read()

#Apply the preprocessing to the loaded dataset
content = preprocessing(content)

# Split the dataset into multiple small chunks
splitter = MarkdownTextSplitter(chunk_size = 200, chunk_overlap = 100)
chunks = splitter.split_text(content)

# Vectorize the chunks/content
emd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = emd.embed_documents(chunks)

# Store the vectors
vectore_store = Chroma.from_texts(collection_name="my_vector_data",texts=chunks, embedding=emd)

# Code for retriving first 3 relevant chunks
retriever = vectore_store.as_retriever(search_kwargs={"k": 3})

# Connecting to the local ollama model
llm = ChatOllama(
    model="llama3",
    base_url="http://localhost:11434"
)

# Basic promt for the formatting of the answers
qa_prompt = ChatPromptTemplate.from_template("""
Use the context to answer the question clearly and professionally.

Always format the answer using:
- Headings
- Bullet points
- New lines
- Short and readable paragraphs
- Markdown formatting

If the question asks for:
- Differences → Use bullet points or a table.
- Definitions → Use short sections.
- Explanations → Use clean step-by-step points.

Context:
{context}

Question:
{question}

Answer:
""")

# Loading the model adn building the chain
def qa_chain_():
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type= "stuff",  # simplest
        chain_type_kwargs={"prompt": qa_prompt}
    )
    return qa_chain