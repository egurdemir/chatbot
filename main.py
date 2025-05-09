import streamlit as st
import os
import tempfile
import numpy as np 

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="llama3" , base_url="http://localhost:11434")


# --- Prompt Template ---
TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Question: {question} 
Context: {context} 
Answer:
"""

# --- Document Loading Function ---
def load_document(file_path):
    """Loads a document from a file path."""
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return None


# --- Text Splitting Function ---
def split_text(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    data = text_splitter.split_documents(documents)
    return data


# --- Indexing Function ---
def index_documents(documents):
    """Adds documents to the vector store."""
    vector_store.add_documents(documents)


# --- Retrieval Function ---
def retrieve_documents(query):
    """Retrieves similar documents from the vector store."""
    return vector_store.similarity_search(query)


# --- Question Answering Function ---
def answer_question(question, context):
    """Answers a question based on retrieved context."""
    prompt = PromptTemplate.from_template(TEMPLATE)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


# --- Streamlit App ---
st.set_page_config(page_title="PDF Chatbot", page_icon="🧠")
st.title("PDF Question Answering Chatbot with RAG and DeepSeek-R1")

st.markdown(
    """
    This chatbot allows you to ask questions about a PDF you upload.
    It utilizes a Retrieval-Augmented Generation (RAG) system powered by the DeepSeek-R1 model for enhanced reasoning and problem-solving capabilities.
    """
)

# --- Sidebar for Examples ---
with st.sidebar:
    st.header("Example Questions")
    example_questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the conclusions presented in this document?",
        "What are some of the key concepts discussed here?",
    ]
    for example in example_questions:
        if st.button(example, key=example):
            st.session_state.example_question = example



uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:

    with open("temp_file.pdf", "wb") as f:
      f.write(uploaded_file.read())
    file_path = "temp_file.pdf"
    
    with st.spinner("Loading and processing document..."):
        documents = load_document(file_path)
        if documents:
          chunked_documents = split_text(documents)
          index_documents(chunked_documents)
    

    if "messages" not in st.session_state:
      st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
          st.markdown(message["content"])
    
    
    if 'example_question' in st.session_state:
      question = st.session_state.example_question
      st.session_state.messages.append({"role": "user", "content": question})
      with st.chat_message("user"):
          st.markdown(question)

      with st.chat_message("assistant"):
          with st.spinner("Processing..."):
              retrieved_docs = retrieve_documents(question)
              context = "\n\n".join([doc.page_content for doc in retrieved_docs])
              answer = answer_question(question, context)
              st.markdown(answer)
              st.session_state.messages.append({"role": "assistant", "content": answer})
      del st.session_state.example_question 

    if question := st.chat_input("Ask a question about the document:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                retrieved_docs = retrieve_documents(question)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                answer = answer_question(question, context)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

   
    os.remove(file_path)
