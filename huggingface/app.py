from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

loader = PyPDFDirectoryLoader("./pdf_files")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_documents = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

## Create a FAISS vector store from the document chunks
vectorstore = FAISS.from_documents(splitted_documents[:100], embeddings)

## Query using Similarity Search
query="WHAT IS HEALTH INSURANCE COVERAGE?"
relevant_docments=vectorstore.similarity_search(query)

print(relevant_docments[0].page_content)

retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
print(retriever)

# Replace the HuggingFacePipeline.from_model_id with this:
model_name = "google/flan-t5-base"  # Using base model for faster loading, you can use xxl if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=300,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)

query="What is the health insurance coverage?"
response=llm.invoke(query)
print(response)

# Update the prompt template to use the correct variable names expected by RetrievalQA
prompt_template = """
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

Context: {context}

Question: {question}

Answer:
"""

# Remove input_variables as they're automatically detected
prompt = PromptTemplate.from_template(prompt_template)

# Update the RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# The query should be passed as a dictionary with the key "query"
query = "What is the health insurance coverage?"
response = retrieval_qa.invoke({"query": query})
print(response["result"])
print(response["source_documents"])