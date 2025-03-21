from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

loader = PyPDFLoader("Here are some common SQL interview questions along with their answers.pdf")
docs = loader.load()
print(docs[0])