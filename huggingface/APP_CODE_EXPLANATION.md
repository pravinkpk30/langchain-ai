# Code Explanation: app.py

This document provides a detailed, line-by-line explanation of the script that implements a document Q&A system using LangChain and HuggingFace's T5 model.

## Table of Contents
1. [Imports](#1-imports)
2. [Environment Setup](#2-environment-setup)
3. [Document Loading](#3-document-loading)
4. [Text Splitting](#4-text-splitting)
5. [Embeddings Setup](#5-embeddings-setup)
6. [Vector Store Creation](#6-vector-store-creation)
7. [Model Initialization](#7-model-initialization)
8. [Prompt Template](#8-prompt-template)
9. [Retrieval QA Chain](#9-retrieval-qa-chain)
10. [Query Execution](#10-query-execution)
11. [Key Concepts](#11-key-concepts)

## 1. Imports

```python
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
```

- **PyPDFLoader/PyPDFDirectoryLoader**: For loading PDF documents
- **RecursiveCharacterTextSplitter**: Splits documents into chunks with overlap
- **HuggingFaceEmbeddings**: Generates embeddings using HuggingFace models
- **HuggingFacePipeline**: Wrapper for HuggingFace models in LangChain
- **AutoModelForSeq2SeqLM/AutoTokenizer**: For loading T5 models and tokenizers
- **FAISS**: Efficient similarity search library
- **PromptTemplate**: For creating structured prompts
- **RetrievalQA**: Chain for question-answering with retrieval
- **dotenv**: For loading environment variables
- **os**: For file operations

## 2. Environment Setup

```python
load_dotenv()
```

- Loads environment variables from a `.env` file
- Used for configuration management and API keys

## 3. Document Loading

```python
loader = PyPDFDirectoryLoader("./pdf_files")
documents = loader.load()
```

- Loads all PDF files from the `./pdf_files` directory
- `documents` contains the loaded PDF content

## 4. Text Splitting

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_documents = splitter.split_documents(documents)
```

- Splits documents into chunks of 1000 characters with 200 characters overlap
- Ensures context is maintained between chunks

## 5. Embeddings Setup

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

- Initializes the embedding model
- Uses the `all-MiniLM-L6-v2` sentence transformer
- Configures to run on CPU and normalize embeddings

## 6. Vector Store Creation

```python
vectorstore = FAISS.from_documents(splitted_documents[:100], embeddings)
```

- Creates a FAISS vector store from the first 100 document chunks
- Enables efficient similarity search

## 7. Similarity Search Example

```python
query = "WHAT IS HEALTH INSURANCE COVERAGE?"
relevant_documents = vectorstore.similarity_search(query)
print(relevant_documents[0].page_content)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
```

- Performs a similarity search on the vector store
- Retrieves the most relevant document chunk
- Creates a retriever for use in the QA chain

## 8. Model Initialization

```python
model_name = "google/flan-t5-base"
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
```

- Loads the T5-base model and tokenizer
- Sets up a text generation pipeline
- Configures generation parameters (max length and temperature)
- Wraps the pipeline in a LangChain-compatible interface

## 9. Prompt Template

```python
prompt_template = """
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

Context: {context}

Question: {question}

Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)
```

- Defines the structure for the LLM prompt
- Uses placeholders for context and question
- Creates a reusable template

## 10. Retrieval QA Chain

```python
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
```

- Creates a QA chain that combines retrieval and generation
- Uses the "stuff" chain type for simplicity
- Configures the retriever and prompt template
- Enables returning source documents for reference

## 11. Query Execution

```python
query = "What is the health insurance coverage?"
response = retrieval_qa.invoke({"query": query})
print(response["result"])
print(response["source_documents"])
```

- Executes a query against the QA system
- Prints both the generated answer and source documents
- Demonstrates the end-to-end functionality

## Key Concepts

1. **Document Processing Pipeline**:
   - Load → Split → Embed → Store → Retrieve

2. **Retrieval-Augmented Generation (RAG)**:
   - Combines document retrieval with language model generation
   - Provides more accurate and grounded responses

3. **Vector Similarity Search**:
   - Uses FAISS for efficient nearest-neighbor search
   - Enables finding relevant document chunks quickly

4. **Prompt Engineering**:
   - Structures the input to guide the model's responses
   - Includes context and question in a clear format

5. **Model Configuration**:
   - T5 model for sequence-to-sequence tasks
   - Configurable parameters for generation quality

## Usage Notes

1. Place PDF files in a `pdf_files` directory
2. Install required packages:
   ```bash
   pip install langchain-community transformers faiss-cpu sentence-transformers python-dotenv PyPDF2
   ```
3. Run the script:
   ```bash
   python app.py
   ```

This implementation provides a foundation for building document-based Q&A systems and can be extended with additional features like a web interface or support for more document types.

## I've created a comprehensive APP_CODE_EXPLANATION.md file that provides a detailed breakdown of 
app.py
. Here's what it includes:

## Structured Breakdown:
11 main sections following the script's logical flow
Clear section headers for easy navigation
Detailed explanations of each component

## Key Components:
Document loading and processing
Text splitting and embedding
Vector store creation with FAISS
T5 model initialization
Prompt engineering
Retrieval-augmented generation

## Educational Value:
Explains the purpose of each import
Details configuration parameters
Describes the document processing pipeline
Covers key concepts like RAG and vector search

## Practical Information:
Usage instructions
Installation requirements
Example queries
Expected outputs

## The document serves as both a learning resource and a reference for understanding or modifying the script. It's particularly useful for:

- Understanding the document Q&A pipeline
- Debugging or extending the code
- Learning about LangChain and HuggingFace integration
- Implementing similar functionality in other projects