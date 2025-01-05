# Import necessary modules
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Chat model and embeddings
from langchain_community.vectorstores import FAISS  # Vector store for embeddings
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, WebBaseLoader  # Load CSV, PDFs, and web pages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import pyttsx3  # For text-to-speech
import os
import logging
import streamlit as st
import time  # For adding delay between web requests

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Configure logging
logging.basicConfig(filename="chatbot_logs.txt", level=logging.INFO)

# Initialize OpenAI chat LLM and text-to-speech engine
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
engine = pyttsx3.init()

# Create a custom prompt for professional responses
custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="You are an educational expert providing information on NWSISD. Answer this professionally: {question}"
)

# Load documents from CSV and PDF
csv_loader = CSVLoader("documents/NWSISD_Schools.csv")
pdf_loader = PyPDFLoader("documents/NWSISD_Report.pdf")

# List of all school-related URLs
urls = [
    "https://nwsisd.edu/northwest-high",
    "https://nwsisd.edu/lakeside-elementary",
    "https://nwsisd.edu/pine-grove-middle",
    "https://nwsisd.edu/summit-academy",
    "https://nwsisd.edu/district-news",
    "https://nwsisd.edu/announcements",
    "https://nwsisd.edu/school-calendar",
    "https://nwsisd.edu/pta-updates",
    "https://nwsisd.edu/school-performance-reports"
]

# Load content from all URLs
web_documents = []
for url in urls:
    web_loader = WebBaseLoader(url)
    try:
        web_documents += web_loader.load()
        time.sleep(1)  # Add delay between requests to prevent rate-limiting
    except Exception as e:
        logging.error(f"Failed to load {url}: {e}")

# Combine all documents
documents = csv_loader.load() + pdf_loader.load() + web_documents

# Split documents into chunks and create embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
embedding = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embedding)

# Create a conversational retrieval chain with memory
qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_store.as_retriever())

# Streamlit Web Interface
st.title("NWSISD School District Chatbot")
st.write("Ask me anything about the NWSISD School District!")
chat_history = []

user_input = st.text_input("Enter your question:")
if user_input:
    # Get the chatbot's response
    response = qa_chain.invoke({"question": user_input, "chat_history": chat_history})
    bot_answer = response["answer"]

    # Display response
    st.write(f"**Answer:** {bot_answer}")
    chat_history.append((user_input, bot_answer))

    # Text-to-Speech Output
    engine.say(bot_answer)
    engine.runAndWait()

    # Log the conversation
    logging.info(f"User: {user_input} | Bot: {bot_answer}")

    # Feedback Mechanism
    feedback = st.radio("Was this helpful?", ("Yes", "No"))
    if feedback == "No":
        st.write("Thank you for your feedback! We will improve.")
