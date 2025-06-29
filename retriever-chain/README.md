# LangChain Retriever Chain with Google Gemini

A powerful Retrieval-Augmented Generation (RAG) application that uses LangChain and Google's Gemini model to answer questions based on PDF documents. This implementation demonstrates how to create a document-based question-answering system with vector similarity search.

## Features

- **Document Processing**: Load and process PDF documents
- **Text Chunking**: Split documents into manageable chunks with overlap
- **Vector Embeddings**: Generate embeddings using HuggingFace's sentence transformers
- **Similarity Search**: FAISS for efficient vector similarity search
- **Interactive UI**: Streamlit-based web interface for easy interaction
- **Google Gemini Integration**: Leverages Google's powerful Gemini model for generating responses

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Google Gemini API key
- Required Python packages (see Installation)

## Installation

1. Clone the repository (if not already done):
   ```bash
   git clone <repository-url>
   cd langchain-ai/retriever-chain

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Google Gemini API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Running the Application

1. Navigate to the retriever-chain directory:
   ```bash
   cd retriever-chain
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run retriever.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

## Monitoring with LangSmith

This project is integrated with LangSmith for monitoring and observability. To use LangSmith:

1. Create a LangSmith account and API key (for monitoring)
2. Set the following environment variables:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=your_project_name
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run retriever.py
   ```
## How it works

1. The application uses LangChain's `SequentialChain` to create two connected chains:
   - The first chain loads and processes the PDF document using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
   - The second chain creates embeddings for the document chunks using `HuggingFaceEmbeddings` and stores them in a FAISS vector store.
   - The third chain creates a retrieval chain using the vector store and a `ChatGoogleGenerativeAI` model.
   - The final chain combines the retrieval chain with a `ChatGoogleGenerativeAI` model to generate responses based on the user's query.

2. The application also includes a Streamlit web interface that allows users to upload a PDF document and ask questions about it.

3. The application also includes a Streamlit web interface that allows users to upload a PDF document and ask questions about it.

4. The application also includes a Streamlit web interface that allows users to upload a PDF document and ask questions about it.

### Document Processing

1. The application uses LangChain's `PyPDFLoader` to load the PDF document.
2. The application uses LangChain's `RecursiveCharacterTextSplitter` to split the document into smaller chunks.
3. The application uses LangChain's `HuggingFaceEmbeddings` to create embeddings for the document chunks.
4. The application uses LangChain's `FAISS` to create a vector store from the document chunks.

### Text Chunking

1. The application uses LangChain's `RecursiveCharacterTextSplitter` to split the document into smaller chunks.
2. The application uses LangChain's `RecursiveCharacterTextSplitter` to split the document into smaller chunks.
3. The application uses LangChain's `RecursiveCharacterTextSplitter` to split the document into smaller chunks.
4. The application uses LangChain's `RecursiveCharacterTextSplitter` to split the document into smaller chunks.

### Vector Embeddings
1. Text chunks are converted into vector embeddings using HuggingFace's sentence transformers
2. FAISS (Facebook AI Similarity Search) is used to store and efficiently search these embeddings

### Question Answering
1. When a question is asked, the system retrieves the most relevant document chunks
2. The retrieved context is combined with the question
3. Google's Gemini model generates a response based on the provided context

## Project Structure

```
retriever-chain/
├── retriever.py     # PDF document processing
├── requirements.txt # Python dependencies
└── .env            # Environment variables
```

## Troubleshooting

- If you encounter any issues, please check the error message and try to resolve it.
- If you are still having issues, please contact the author for assistance.

## License

MIT License

## Author

Praveen Kumar
