from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
llm = GoogleGenerativeAI(model = "gemini-2.0-flash")

loader = PyMuPDFLoader("SQL Revision Notes.pdf")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=150)
docs = text_splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(docs,embeddings)

query = "What is primary key?"
docs = db.similarity_search(query,k=1)

for i in docs:
    print(i.page_content.replace("\n"," "))
    print("-----------------")

