import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
import time

# This application demonstrates a basic RAG (Retrieval-Augmented Generation) pipeline, combining document retrieval with an LLM for question answering.

load_dotenv()

llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192", temperature=0.2)

if "vectors" not in st.session_state:
    st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


# Create a prompt template
prompt_template = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# streamlit ui
st.title("Chat with Groq Demo")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    with st.spinner("Thinking..."):
        start=time.process_time()
        response=retriever_chain.invoke({"input": prompt})
        print("Time taken for retrieval: ", time.process_time() - start)
        st.write(response["answer"])
        end=time.process_time()
        print("Time taken for response: ", end - start)
    
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
        
        
