import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Set page config
st.set_page_config(page_title="Document Q&A with T5", page_icon="📄")

# Title and description
st.title("📄 Document Q&A with T5")
st.write("Ask questions about the loaded document and get AI-powered answers!")

# Initialize session state for the model and retriever
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
    st.session_state.retriever = None
    st.session_state.qa_chain = None

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

def initialize_model(model_name, max_length, temperature):
    """Initialize the T5 model and tokenizer"""
    with st.spinner(f"Loading {model_name}..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

def setup_retrieval_qa(llm, documents):
    """Set up the retrieval QA system"""
    # Split documents
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

# Main app
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

if __name__ == "__main__":
    main()