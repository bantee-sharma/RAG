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

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)

chunk = text_splitter.split_documents(docs)
embedd = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(chunk,embedd)

retriever = vector_store.as_retriever(search_type="similarity", kwargs={'k':3})

prompt = PromptTemplate(
    template= '''You are a helpfull AI assistant.
    Answer the question from the following context.
    If context is insufficient just say, I don't know.
    if anyone asked quesion in english then give answer in Englsih.
    {context}
    Question : {question}''',
    input_variables=["context","question"]
)

question = "What is this documnet about?"

retriever_docs = retriever.invoke(question)
context = " ".join([i.page_content for i in retriever_docs])

final_prompt = prompt.invoke({"context":context,"question":question})

res = llm.invoke(final_prompt)
print(res)
