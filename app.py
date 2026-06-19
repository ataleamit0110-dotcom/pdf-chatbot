import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("📄 PDF Q&A Chatbot")

loader = PyMuPDFLoader("Amit_Atale_AI_Resume.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20
)

docs = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(
    docs,
    embedding_model
)

query = st.text_input("Ask a question about the PDF")

if query:

    results = vector_db.similarity_search(query, k=2)

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    st.write(response.choices[0].message.content)