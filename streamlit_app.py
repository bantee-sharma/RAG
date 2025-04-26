import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure 'data' directory exists
os.makedirs("data", exist_ok=True)

# Streamlit UI
st.title("📄 AI-Powered PDF Q&A Chatbot")
st.write("Upload a PDF and ask questions about its content.")

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("data", uploaded_file.name)

    # Save uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Load and process PDF
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Split document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(documents)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings()

    # Store in FAISS vector DB
    db = FAISS.from_documents(doc_chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Define LLM and Retrieval QA Chain
    llm = GoogleGenerativeAI(model="gemini-2.0-flash")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # User input for query
    query = st.text_input("🔍 Ask a question about the PDF:")
    if query:
        response = qa_chain.run(query)
        st.write("💡 **Answer:**", response)
