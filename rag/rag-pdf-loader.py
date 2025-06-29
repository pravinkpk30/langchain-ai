# Import necessary libraries
from langchain.embeddings import HuggingFaceEmbeddings  # For generating text embeddings
from langchain.vectorstores import FAISS  # For vector similarity search
from langchain.chains import RetrievalQA  # For QA with retrieval
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini model
from dotenv import load_dotenv  # For loading environment variables
import streamlit as st  # For creating web interface
import os  # For file and system operations

# Load environment variables from .env file
load_dotenv()

# Initialize the Google Gemini language model
# Using 'gemini-2.0-flash' model with API key from environment variables
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# Load PDF document using PDFMinerLoader
# This extracts text from the specified PDF file
from langchain.document_loaders import PDFMinerLoader
loader = PDFMinerLoader("attention.pdf")
documents = loader.load()

# Split the document into smaller chunks with overlap
# This helps in processing large documents and maintaining context
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_documents = splitter.split_documents(documents)

# Initialize HuggingFace embeddings
# Using 'all-MiniLM-L6-v2' model which is good for general purpose embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the document chunks
# This enables efficient similarity search
vectorstore = FAISS.from_documents(splitted_documents, embeddings)

# Create a RetrievalQA chain
# This combines the language model with the retriever for question-answering
qa = RetrievalQA.from_chain_type(
    llm=llm,  # The language model to use
    chain_type="stuff",  # Method for combining documents
    retriever=vectorstore.as_retriever()  # The retriever to fetch relevant documents
)

# Set up the Streamlit web interface
st.title("LangChain PDF Based RAG")

# Create a text input field for user's question
topic = st.text_input("Enter your question about the document")

# When user enters a question
if topic:
    # Show a spinner while processing
    with st.spinner("Searching for answers..."):
        # Get the answer from the QA chain and display it
        st.write(qa.run(topic))