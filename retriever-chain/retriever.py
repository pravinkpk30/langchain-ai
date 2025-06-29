# Import necessary libraries
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Load and process the PDF
loader = PyPDFLoader("attention.pdf")  # Make sure this file exists in the same directory
documents = loader.load()

# Split the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_documents = splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splitted_documents, embeddings)

# Create the prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that can answer questions about the document.
Context: {context}
Question: {input}
Please provide a detailed answer based on the context above.
""")

# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the retrieval chain
retriever = vectorstore.as_retriever()
qa = create_retrieval_chain(retriever, document_chain)

# Streamlit UI
st.title("LangChain PDF Based RAG")

topic = st.text_input("Enter your question about the document")

if topic:
    with st.spinner("Searching for answers..."):
        try:
            # Get the response
            response = qa.invoke({"input": topic})
            st.write(response["answer"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try a different question or check your input file.")