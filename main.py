import streamlit as st
import os
import numpy as np

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.prompts import PromptTemplate

# Optional metrics
try:
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer
    metrics_available = True
except ImportError:
    metrics_available = False

# --- Session State Initialization ---
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_log" not in st.session_state:
    st.session_state.qa_log = {}

# --- Ollama Components ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model=st.session_state.selected_model, base_url="http://localhost:11434")

# --- Prompt Template ---
TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Question: {question} 
Context: {context} 
Answer:
"""

# --- Functions ---
def load_document(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return None

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
    return splitter.split_documents(documents)

def index_documents(documents):
    vector_store.add_documents(documents)

def retrieve_documents(query):
    return vector_store.similarity_search(query)

def answer_question(question, context):
    prompt = PromptTemplate.from_template(TEMPLATE)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

def evaluate_answer(reference, candidate):
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    bleu = sentence_bleu([ref_tokens], cand_tokens)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, candidate)
    return {
        "BLEU": round(bleu, 4),
        "ROUGE-1 F1": round(rouge['rouge1'].fmeasure, 4),
        "ROUGE-L F1": round(rouge['rougeL'].fmeasure, 4),
    }

# --- Streamlit App ---
st.set_page_config(page_title="PDF Chatbot", page_icon="🧠")
st.title("PDF Question Answering Chatbot with LLaMa 3, Gemma 3, and DeepSeek R1")
st.write("Upload a PDF document and ask questions about its content. Compare answers from different models.")

# --- Sidebar ---
with st.sidebar:
    st.header("Choose a Model")
    if st.button("Use LLaMA 3"):
        st.session_state.selected_model = "llama3"
    if st.button("Use Gemma 3"):
        st.session_state.selected_model = "gemma3"
    if st.button("Use DeepSeek R1"):
        st.session_state.selected_model = "deepseek-r1"
    st.markdown(f"**Current model:** `{st.session_state.selected_model}`")

    st.header("Example Questions")
    examples = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the conclusions presented in this document?",
        "What are some of the key concepts discussed here?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.example_question = ex

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("temp_file.pdf", "wb") as f:
        f.write(uploaded_file.read())
    file_path = "temp_file.pdf"

    with st.spinner("Loading and processing document..."):
        documents = load_document(file_path)
        if documents:
            chunks = split_text(documents)
            index_documents(chunks)

    # --- Display chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                model_tag = msg.get("model", "unknown")
                st.markdown(f"**({model_tag})**: {msg['content']}")
            else:
                st.markdown(msg["content"])

    # --- Process example question ---
    if 'example_question' in st.session_state:
        question = st.session_state.example_question
        del st.session_state.example_question

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                context = "\n\n".join([doc.page_content for doc in retrieve_documents(question)])
                answer = answer_question(question, context)
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "model": st.session_state.selected_model
                })

                if question:
                    q_key = question.strip().lower()
                    if q_key not in st.session_state.qa_log:
                        st.session_state.qa_log[q_key] = {}
                    st.session_state.qa_log[q_key][st.session_state.selected_model] = answer.strip()

    # --- Process manual question input ---
    if question := st.chat_input("Ask a question about the document:"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                context = "\n\n".join([doc.page_content for doc in retrieve_documents(question)])
                answer = answer_question(question, context)
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "model": st.session_state.selected_model
                })

                if question:
                    q_key = question.strip().lower()
                    if q_key not in st.session_state.qa_log:
                        st.session_state.qa_log[q_key] = {}
                    st.session_state.qa_log[q_key][st.session_state.selected_model] = answer.strip()

    # --- Evaluation UI ---
    with st.expander("🧪 Evaluate Model Answers with BLEU / ROUGE"):
        reference_answer = st.text_area("Enter the reference answer for the last question:", key="ref_input")
        q_eval_key = list(st.session_state.qa_log.keys())[-1] if st.session_state.qa_log else None

        if reference_answer and q_eval_key:
            if not metrics_available:
                st.error("Please install `nltk` and `rouge-score` packages.")
            else:
                st.markdown("### 📊 Model Evaluation Scores")
                for mname, moutput in st.session_state.qa_log[q_eval_key].items():
                    scores = evaluate_answer(reference_answer, moutput)
                    st.markdown(f"**Model: `{mname}`**")
                    st.json(scores)

    os.remove(file_path)
