# Code Explanation: app_mvp.py

This document provides a detailed, line-by-line explanation of the Document Q&A application.

## Table of Contents
1. [Imports](#1-imports)
2. [Streamlit UI Setup](#2-streamlit-ui-setup)
3. [Session State Initialization](#3-session-state-initialization)
4. [Model Configuration Sidebar](#4-model-configuration-sidebar)
5. [Model Initialization](#5-model-initialization)
6. [Retrieval QA Setup](#6-retrieval-qa-setup)
7. [Main Application Logic](#7-main-application-logic)
8. [Execution](#8-execution)

## 1. Imports

```python
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
```

- **streamlit**: Web application framework for creating the UI
- **RecursiveCharacterTextSplitter**: Splits documents into chunks with overlap
- **HuggingFaceEmbeddings**: Generates embeddings using HuggingFace models
- **HuggingFacePipeline**: Wrapper for HuggingFace models in LangChain
- **AutoModelForSeq2SeqLM, AutoTokenizer**: For loading T5 models and tokenizers
- **FAISS**: Efficient similarity search library
- **PromptTemplate**: For creating structured prompts
- **RetrievalQA**: Chain for question-answering with retrieval
- **os**: For file operations

## 2. Streamlit UI Setup

```python
# Set page config
st.set_page_config(page_title="Document Q&A with T5", page_icon="📄")

# Title and description
st.title("📄 Document Q&A with T5")
st.write("Ask questions about the loaded document and get AI-powered answers!")
```

- Sets up the page title and icon in the browser tab
- Displays a title and brief description on the page

## 3. Session State Initialization

```python
# Initialize session state for the model and retriever
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
    st.session_state.retriever = None
    st.session_state.qa_chain = None
```

- Uses Streamlit's session state to persist data between reruns
- Tracks whether the model is initialized and stores the retriever and QA chain

## 4. Model Configuration Sidebar

```python
# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")
    model_name = st.selectbox(
        "Select Model",
        ["google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl"],
        index=0
    )
    max_length = st.slider("Max Response Length", 50, 500, 200, 50)
    temperature = st.slider("Temperature", 0.1, 1.0, 0.2, 0.1)
```

- Creates a collapsible sidebar
- Allows users to select different T5 model sizes
- Configures response length and temperature parameters

## 5. Model Initialization

```python
def initialize_model(model_name, max_length, temperature):
    """Initialize the T5 model and tokenizer"""
    with st.spinner(f"Loading {model_name}..."):
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create text generation pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature
        )
        
        # Wrap in LangChain interface
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
```

- Loads the specified T5 model and tokenizer
- Sets up a text generation pipeline with configurable parameters
- Wraps the pipeline in a LangChain-compatible interface

## 6. Retrieval QA Setup

```python
def setup_retrieval_qa(llm, documents):
    """Set up the retrieval QA system"""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create prompt template
    prompt_template = """
    Use the following piece of context to answer the question asked.
    Please try to provide the answer only based on the context

    Context: {context}

    Question: {question}

    Answer:
    """
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain
```

- Splits documents into manageable chunks with overlap
- Generates embeddings for the document chunks
- Creates a FAISS vector store for efficient similarity search
- Defines a prompt template for the QA task
- Sets up the retrieval-augmented generation chain

## 7. Main Application Logic

```python
def main():
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open("temp_file", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize model and QA chain if not already done
        if not st.session_state.model_initialized:
            with st.spinner("Initializing model and setting up document processing..."):
                llm = initialize_model(model_name, max_length, temperature)
                
                # Load and process the uploaded file
                if uploaded_file.name.endswith('.pdf'):
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader("temp_file")
                else:  # txt file
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader("temp_file")
                
                documents = loader.load()
                st.session_state.qa_chain = setup_retrieval_qa(llm, documents)
                st.session_state.model_initialized = True
        
        # Chat interface
        st.subheader("Ask a question about the document")
        user_question = st.text_input("Your question:", key="user_question")
        
        if user_question:
            with st.spinner("Thinking..."):
                response = st.session_state.qa_chain.invoke({"query": user_question})
                
                # Display answer
                st.subheader("Answer:")
                st.write(response["result"])
                
                # Display source documents
                with st.expander("Source Documents"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.write(f"**Source {i+1}** (Page {doc.metadata.get('page', 'N/A')}):")
                        st.text(doc.page_content)
                        st.write("---")
    else:
        st.info("Please upload a document to get started.")
```

- Handles file uploads (PDF or text)
- Initializes the model and document processing on first upload
- Provides a text input for user questions
- Displays answers and source documents
- Shows loading states and error messages

## 8. Execution

```python
if __name__ == "__main__":
    main()
```

- Standard Python idiom to run the main function when the script is executed directly
- Ensures the code doesn't run when imported as a module

## Key Concepts

1. **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with language model generation
2. **Vector Similarity Search**: Uses FAISS for efficient document retrieval
3. **Chunking**: Splits documents into manageable pieces for processing
4. **Prompt Engineering**: Structures the input to guide the model's responses
5. **Streamlit State Management**: Maintains state between user interactions

This implementation provides a solid foundation that can be extended with additional features like conversation history, document summarization, or support for more file types.

I've created a comprehensive CODE_EXPLANATION.md file that breaks down the 
app_mvp.py
 code in detail. 

# Here's what it includes:

## 1. Structured Breakdown:
- Organized into 8 main sections following the code's logical flow
- Each section explains a specific component of the application

## 2. Detailed Explanations:
- Line-by-line analysis of key code blocks
- Purpose and functionality of each import
- Explanation of Streamlit UI components
- Model initialization and configuration
- Document processing and retrieval setup

## 3. Key Concepts:
- Retrieval-Augmented Generation (RAG)
- Vector similarity search with FAISS
- Document chunking and embedding
- Streamlit state management
- Prompt engineering

## 4. Visual Aids:
- Code blocks with syntax highlighting
- Clear section headers
- Logical flow between components

## 5. The document serves as both a learning resource and future reference for understanding or modifying the application. It's particularly useful for:

- New developers joining the project
- Debugging and maintenance
- Extending functionality
- Understanding the architecture
