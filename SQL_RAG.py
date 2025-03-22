from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
load_dotenv()
llm = GoogleGenerativeAI(model = "gemini-2.0-flash")

loader = PyPDFLoader("SQL Revision Notes.pdf")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=150)
docs = text_splitter.split_documents(document)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(docs,embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

prompt = PromptTemplate(
    template= "You are an expert AI assitant. Answer the based on provide document:\n {context} question:{query} \n Answer: ",
    input_variables=["context","query"]
)

qa_chain = RetrievalQA.from_chain_type(llm,retriever = retriever)

def ask_question(query):
    response = qa_chain.run(query)
    return response

query = "What is the primary key?"
ans = ask_question(query)

print(ans)

