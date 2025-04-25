import streamlit as st
import os
import tempfile
import faiss
import numpy as np 

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

st.set_page_config(page_title="🧠 PDF Chatbot", layout="wide")
st.title(" Q&A Chatbot with Multiple PDFs")

uploaded_files = st.file_uploader("Upload one or more PDF's:", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing..."):
        all_docs = []

    
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            all_docs.extend(pages)

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(all_docs)

        embeddings = OllamaEmbeddings(model="llama3", base_url="http://localhost:11434")

        single_vector = embeddings.embed_query("some text data")
        embedding_dim = len(single_vector)  

        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        
        ids = vector_store.add_documents(documents=chunks)

        prompt = PromptTemplate.from_template("""
        You are an expert assistant based on the content of a PDF document. 
        Answer the user's question using only the information from the document. 
        If the answer is not in the document, say "I don't know based on the provided information."

        Documents:
        {context}

        Question: {question}

        Answer:
        """)

        
        llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

        
        st.markdown("### ❓ You can ask question:")
        query = st.text_input("Your question:")
        if query:
            with st.spinner("Looking for answer..."):
                result = chain.run(query)
                st.markdown(f"**Answer:** {result}")