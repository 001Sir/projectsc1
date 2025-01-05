# Import necessary modules
from langchain.chat_models import ChatOpenAI  # Chat model
from langchain.embeddings.openai import OpenAIEmbeddings  # Embeddings
from langchain.vectorstores import FAISS, Chroma  # Added Chroma for persistent storage
from langchain.document_loaders import CSVLoader, PyPDFLoader, WebBaseLoader  # Load CSV, PDFs, and web pages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from deep_translator import GoogleTranslator
from rapidfuzz import fuzz
import os
import logging
import streamlit as s
import random
import streamlit as st
import time  # For adding delay between web requests
import glob  # For automatically listing all files
import pandas as pd  # To read CSV files and count schools
from concurrent.futures import ThreadPoolExecutor  # For faster web loading

# Install dependencies if needed
try:
    from rapidfuzz import fuzz, process  # For fuzzy matching
except ImportError:
    os.system('pip install rapidfuzz')
    from rapidfuzz import fuzz, process

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Configure logging
logging.basicConfig(filename="chatbot_logs.txt", level=logging.INFO)

# Ensure chat history is initialized
chat_history = []
selected_language = "English"

# Initialize OpenAI chat model with memory and custom prompt
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=500)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

custom_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an educational expert who provides detailed, summarized, and clear answers about the NWSISD School District. "
        "The list of schools, their levels, and programs is available, and you know how many schools exist. "
        "When asked, always mention the number of schools and relevant details if available. Answer this: {question}"
    )
)

# Automatically load all CSV and PDF files in the "documents" folder
csv_files = glob.glob("documents/*.csv")
pdf_files = glob.glob("documents/*.pdf")

csv_documents = []
pdf_documents = []

# List of schools for fuzzy matching
school_names = [
    "Birch Grove Elementary School for the Arts",
    "Rogers Elementary STEM Magnet School",
    "R.L. Stevenson Elementary School - An IB World School",
    "Tatanka Elementary STEM School",
    "University Avenue ACES: Aerospace, Children's Engineering, and Science",
    "Weaver Lake Elementary: A Science, Math, and Technology School",
    "Zanewood Community School: STEAM",
    "Anoka Middle School for the Arts",
    "Brooklyn Center STEAM Middle School",
    "Brooklyn Middle STEAM Magnet School",
    "Fridley Middle School - An IB World School",
    "Rockford Middle School - Center for Environmental Studies",
    "Salk Middle School Pre-Engineering STEM Magnet",
    "Anoka High School Center for Science, Technology, Engineering, Arts, and Math",
    "Blaine High School Center for Engineering, Mathematics & Science (CEMS)",
    "Brooklyn Center STEAM High School",
    "Coon Rapids High School Center for Biomedical Sciences & Engineering",
    "Fridley High School - An IB World School",
    "Osseo Senior High Health Science Magnet Program",
    "Park Center Senior High - An IB World School",
    "Rockford High School - An IB World School",
    "Brooklyn Center Elementary STEAM School",
    "Hayes Elementary School - An IB World School",
    "Evergreen Park STEM School of Innovation",
    "Montrose Elementary School of Innovation",
    "Rockford Elementary Arts Magnet School",
    "Summit Academy",
    "Lakeside Elementary School"
]

# File-based keyword routing
file_mapping = {
    "general": "documents/general_info.csv",
    "contact": "documents/contact_info.csv",
    "programs": "documents/academic_programs.csv",
    "extracurricular": "documents/extracurricular_activities.csv",
    "facilities": "documents/facilities.csv",
    "enrollment": "documents/enrollment.csv",
    "policies": "documents/school_policies.pdf",
    "pta": "documents/pta_info.csv",
    "events": "documents/events_calendar.csv",
    "safety": "documents/health_and_safety.pdf",
    "performance": "documents/school_performance.pdf",
    "special": "documents/special_programs.csv",
    "fees": "documents/tuition_fees.csv"
}

keywords = {
    "general": ["principal", "hours", "grade levels", "dress code", "uniform"],
    "contact": ["contact", "email", "phone", "address", "admissions", "transfers"],
    "programs": ["program", "curriculum", "special education", "AP", "IB", "dual enrollment"],
    "extracurricular": ["sports", "clubs", "band", "choir", "theater"],
    "facilities": ["library", "cafeteria", "transportation", "playground"],
    "enrollment": ["admission", "requirements", "apply", "enrollment period"],
    "policies": ["attendance", "discipline", "homework"],
    "pta": ["pta", "parent-teacher association", "volunteer"],
    "events": ["conference", "holiday", "event"],
    "safety": ["safety", "protocol", "mental health", "counselor"],
    "performance": ["test scores", "graduation rate", "rank"],
    "special": ["gifted", "ESL", "after-school", "summer"],
    "fees": ["tuition", "fees", "financial aid"]
}

# Fuzzy matching function for detecting school names
def detect_school_name(user_input):
    best_match = process.extractOne(user_input, school_names, scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] >= 75:
        return best_match[0]
    return None

# Fuzzy matching for file category detection
def get_related_document(user_input):
    for category, synonyms in keywords.items():
        best_match = process.extractOne(user_input.lower(), synonyms, scorer=fuzz.partial_ratio)
        if best_match and best_match[1] >= 70:
            return file_mapping.get(category)
    return None

# Load all CSV files and count schools
num_schools = 0
for csv_file in csv_files:
    try:
        csv_loader = CSVLoader(file_path=csv_file)
        csv_documents += csv_loader.load()
        df = pd.read_csv(csv_file)
        if "School Name" in df.columns:
            num_schools += len(df)
        logging.info(f"Loaded CSV file: {csv_file} with {len(df)} entries.")
    except Exception as e:
        logging.error(f"Failed to load CSV file {csv_file}: {e}")

# Load all PDF files
for pdf_file in pdf_files:
    try:
        pdf_loader = PyPDFLoader(file_path=pdf_file)
        pdf_documents += pdf_loader.load()
        logging.info(f"Loaded PDF file: {pdf_file}")
    except Exception as e:
        logging.error(f"Failed to load PDF file {pdf_file}: {e}")

# Combine all documents
documents = csv_documents + pdf_documents

# Split documents into chunks and create embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
embedding = OpenAIEmbeddings()

# Persistent vector store for faster loading
vector_store_path = "vector_store"
if os.path.exists(vector_store_path):
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embedding)
else:
    vector_store = Chroma.from_documents(docs, embedding, persist_directory=vector_store_path)
    vector_store.persist()

# Create a conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vector_store.as_retriever(), memory=memory)

# Clean ChatGPT-like UI
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 700px;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        background-color: #f7f7f8;
    }
    input {
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #ddd;
    }
    .css-1inwz65, .css-q8sbsg {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# ChatGPT-like UI setup
st.markdown("# 🤖 NWSISD School District Assistant")
st.write("Hello! How can I help you today? Also, how has your day been?")

# User input text box
user_input = st.text_input("Your question:")

# Function to detect and suggest schools with improved fuzzy matching
def detect_and_suggest_school(user_input, school_list):
    suggestions = process.extract(user_input, school_list, scorer=fuzz.token_set_ratio, limit=5)
    return [suggestion[0] for suggestion in suggestions if suggestion[1] >= 70]

if user_input:
    # Add a conversational opening
    st.markdown("**🤖 ChatGPT:** Thanks for your question! Let's see what I can find for you.")

    school_name_matches = detect_and_suggest_school(user_input, school_names)

    if not school_name_matches:
        st.write("It seems you haven't mentioned a school name or the name might be misspelled. No worries! Are you asking about the district in general?")
        filtered_schools = detect_and_suggest_school(user_input, school_names)

        if filtered_schools:
            school_name = st.radio("Please confirm which school you're asking about:", ["General District Info"] + filtered_schools)
        else:
            st.write("I'm really sorry, but I couldn't find any schools matching your query. Can you try rephrasing?")
            school_name = None
    else:
        school_name = school_name_matches[0]  # Select the best match for simplicity

    if school_name:
        related_file = get_related_document(user_input)

        if related_file:
            if not os.path.exists(related_file):
                st.error(f"Hmm, I couldn't find the file: {related_file}. Could you double-check if it's placed in the correct directory?")
            else:
                if related_file.endswith(".csv"):
                    try:
                        df = pd.read_csv(related_file)
                        filtered_info = df[df["School Name"].apply(lambda x: fuzz.token_set_ratio(x.lower(), school_name.lower()) >= 70 if pd.notna(x) else False)]
                        if not filtered_info.empty:
                            st.markdown(f"### Here's what I found for **{filtered_info.iloc[0].get('School Name', 'this school')}**")

                            # Provide responses for all columns dynamically, adding conversational tone
                            for column in df.columns:
                                if column != "School Name":
                                    value = filtered_info.iloc[0].get(column, "No information available")
                                    if value != "No information available":
                                        st.write(f"**{column.replace('_', ' ').title()}:** {value}")
                            st.write("Did that answer your question? Let me know if you need more info!")
                        else:
                            st.write("Hmm, I couldn't find matching details for that school. Is there another way I can assist you?")
                    except Exception as e:
                        st.error(f"Oops! There was an error reading the CSV file: {e}")
                elif related_file.endswith(".pdf"):
                    try:
                        pdf_loader = PyPDFLoader(file_path=related_file)
                        pdf_docs = pdf_loader.load()
                        st.markdown("### Here's some content from the PDF")
                        for i, page in enumerate(pdf_docs[:2]):
                            st.write(f"**Page {i + 1}:**")
                            st.write(page.page_content)
                        st.write("Is there anything else you'd like to know? 😄")
                    except Exception as e:
                        st.error(f"Oops! There was an error reading the PDF file: {e}")
        else:
            try:
                if 'chat_history' not in globals():
                    chat_history = []
                response = qa_chain({"question": user_input, "chat_history": chat_history})
                bot_answer = response["answer"]
                if selected_language != "English":
                    bot_answer = translator.translate(bot_answer)
                st.markdown(f"**🤖 ChatGPT:** {bot_answer}")
                st.write("Anything else I can assist with today?")
            except Exception as e:
                st.error(f"Error retrieving answer: {e}")
    else:
        st.write("Please enter or select a valid school name to proceed.")