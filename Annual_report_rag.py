from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader,PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#initialize the llm
llm = GoogleGenerativeAI(model="gemini-2.0-flash")

#load document
loader = PyMuPDFLoader("2022_Annual_Report.pdf")
documents = loader.load()

#split document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap = 150)
docs = text_splitter.split_documents(documents)

# Create embeddings and FAISS vectorstore
embedding = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs,embedding)

# Create a retriever
retriever = db.as_retriever(search_type="similarity",kwargs={"k":1})

#Create Prompt
prompt = PromptTemplate(
    template= "You are an expert AI. Answer the based on document:\n {context} question: {query} \n Answer: ",
    input_variables=["context","query"]
)

#create chain
qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever)

def ask_question(query):
    response = qa_chain.run(query)
    return response

query = "Total revenue in year 2021"
res = ask_question(query)
print("AI Answer: ",res)
