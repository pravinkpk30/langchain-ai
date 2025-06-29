# Import necessary libraries
import streamlit as st  # For creating web interface
import os  # For file and system operations
import bs4  # For web scraping and HTML parsing
from dotenv import load_dotenv  # For loading environment variables
from langchain_google_genai import ChatGoogleGenerativeAI  # Google's Gemini model
from langchain.document_loaders import WebBaseLoader  # For loading web documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text
from langchain.embeddings import HuggingFaceEmbeddings  # For text embeddings
from langchain.vectorstores import Chroma  # For vector storage and retrieval
from langchain.chains import RetrievalQA  # For QA with retrieval

# Load environment variables from .env file
load_dotenv()

# Initialize the Google Gemini language model
# Using 'gemini-2.0-flash' for fast responses with API key from environment variables
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Configure web document loader
# Extracts content from specified web pages, focusing on specific HTML classes
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-title", "post-content", "post-header")
        )
    )
)

# Load and parse the web document
text_documents = loader.load()

# Split the document into smaller chunks with overlap
# This helps in processing large documents and maintaining context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Size of each chunk
    chunk_overlap=200  # Overlap between chunks for context
)
splitted_documents = splitter.split_documents(text_documents)

# Initialize HuggingFace embeddings
# Using 'all-MiniLM-L6-v2' model for efficient text embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create a Chroma vector store from the document chunks
# This enables efficient similarity search and retrieval
vectorstore = Chroma.from_documents(
    documents=splitted_documents,
    embedding=embeddings
)

# Create a RetrievalQA chain
# This combines the language model with the retriever for question-answering
qa = RetrievalQA.from_chain_type(
    llm=llm,  # The language model to use
    chain_type="stuff",  # Method for combining documents
    retriever=vectorstore.as_retriever()  # The retriever to fetch relevant documents
)

# Set up the Streamlit web interface
st.title("LangChain Web Based RAG")

# Create a text input field for user's question
topic = st.text_input("Enter your question about the web content")

# When user enters a question
if topic:
    # Show a spinner while processing
    with st.spinner("Searching the web content for answers..."):
        try:
            # Get the answer from the QA chain and display it
            response = qa.run(topic)
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try a different question or check your internet connection.")