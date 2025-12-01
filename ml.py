import regex as re
import string as sr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import MarkdownTextSplitter


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
    ml_content = f.read()

with open('deep_learning_dataset.md', 'r', encoding="utf-8") as f:
    dl_content = f.read()

with open('AI_dataset.md', 'r', encoding="utf-8") as f:
    ai_content = f.read()

#Apply the preprocessing to the loaded dataset
ml_content = preprocessing(ml_content)
dl_content = preprocessing(dl_content)
ai_content = preprocessing(ai_content)

# Split the dataset into multiple small chunks
splitter = MarkdownTextSplitter(chunk_size = 300, chunk_overlap = 100)
ml_chunks = splitter.split_text(ml_content)
dl_chunks = splitter.split_text(dl_content)
ai_chunks = splitter.split_text(ai_content)

# Vectorize the chunks/content
emd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

ml_vectors = emd.embed_documents(ml_chunks)
dl_vectors = emd.embed_documents(dl_chunks)
ai_vectors = emd.embed_documents(ai_chunks)


# Store the vectors
ml_vectore_store = Chroma.from_texts(collection_name="my_ml_vector_data",texts=ml_chunks, embedding=emd, persist_directory='./db')

dl_vectore_store = Chroma.from_texts(collection_name="my_dl_vector_data",texts=dl_chunks, embedding=emd, persist_directory='./db')

ai_vectore_store = Chroma.from_texts(collection_name="my_ai_vector_data",texts=ai_chunks, embedding=emd, persist_directory='./db')


#Descrition for choosing the file based on the user's query
ml_description = ["Machine learning invloving unsupervised and supervised, reinforcement learning with multiple algorithms such as regression, classification, dimenstionality reduction, evalutation matrix and optimizers"]

dl_description = ["Deep learning with multiple artificial neural networks such as convolutional, recurrent, transformers and backpropagation"]

ai_description = ["artificial intelligence and its different techniques, types of ai, natural language processing and its application, computer vision, application of ai"]

ml_description_emd = emd.embed_documents(ml_description)
dl_description_emd = emd.embed_documents(dl_description)
ai_description_emd = emd.embed_documents(ai_description)

#storing them in a dictionary for further use
embeddings = {
    "ml" : ml_description_emd,
    "dl" : dl_description_emd,
    "ai" : ai_description_emd
}

import joblib
joblib.dump(embeddings, "embedding.pkl")

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

# Loading the model and building the chain
def qa_chain_(retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type= "stuff",  # simplest
        chain_type_kwargs={"prompt": qa_prompt}
    )
    return qa_chain