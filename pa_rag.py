from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

loader = PyMuPDFLoader("PA - Consolidated lecture notes.pdf")
documnet = loader.load()

text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
doc = text_split.split_documents(documnet)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(doc,embeddings)
retriever = db.as_retriever(search_type="similarity",search_kwargs={"k":1})

prompt = PromptTemplate(
    template="You are an expert AI. Answer based on  provide document: \n {context} question: {question}\n Answer: ",
    input_variables=["context","question"]
)

qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever)

def ask_question(query):
    response = qa_chain.invoke(query)
    return response

query = "What are product metrics?"
res = ask_question(query)

print(res)