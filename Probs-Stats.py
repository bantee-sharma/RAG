from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

# Load PDF document
loader = PyPDFLoader("Probs-Stats Revision Notes.pdf")
document = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
doc = text_splitter.split_documents(document)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Ensure a model is specified
db = FAISS.from_documents(doc, embeddings)

# Define retriever
retriever = db.as_retriever(search_type="similarity", kwargs={"k": 5})

# Define prompt
prompt = PromptTemplate(
    template="You are an expert AI assistant. Answer based on the provided document:\n {context} \n Question: {query} \n Answer:",
    input_variables=["context", "query"]
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
    )

# Query the model
query = "What is the p-value?"
response = qa_chain.invoke({"query": query})  # Ensure the input is a dictionary

print(response["result"])  # Extract and print the result
