# LangChain RAG Loaders

This project implements three different Retrieval-Augmented Generation (RAG) loaders using LangChain, each designed to work with different types of documents: PDFs, web pages, and plain text files. The application uses Google's Gemini model for generating responses based on the provided documents.

## Features

- **PDF Document Processing**: Extract and process text from PDF files
- **Web Content Processing**: Scrape and process content from web pages
- **Text File Processing**: Load and process plain text documents
- **Vector Similarity Search**: Efficient document retrieval using FAISS and Chroma
- **Interactive Web Interface**: Built with Streamlit for easy interaction

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Google Gemini API key
- Required Python packages (see [Installation](#installation))

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd langchain-ai/rag

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

5. Run the Streamlit application:
   ```bash
   streamlit run rag-pdf-loader.py
   streamlit run rag-web-loader.py
   streamlit run rag-text-loader.py
   ```
## Project Structure

```
rag/
├── rag-pdf-loader.py     # PDF document processing
├── rag-web-loader.py     # Web content processing
├── rag-text-loader.py    # Text file processing
├── requirements.txt      # Python dependencies
└── .env                 # Environment variables
```

## How It Works

### PDF Loader
- PDFMinerLoader is used to extract text from the PDF file.
- The text is then split into smaller chunks with overlap using TokenTextSplitter.
- HuggingFaceEmbeddings is used to generate embeddings for the text chunks.
- FAISS is used to create a vector store from the document chunks.
- The vector store is then used to retrieve relevant document chunks based on the user's query.
- The retrieved document chunks are then combined with the user's query to generate a response using the Google Gemini model.

### Web Loader
- WebBaseLoader is used to extract text from the web pages.
- The text is then split into smaller chunks with overlap using TokenTextSplitter.
- HuggingFaceEmbeddings is used to generate embeddings for the text chunks.
- Chroma is used to create a vector store from the document chunks.
- The vector store is then used to retrieve relevant document chunks based on the user's query.
- The retrieved document chunks are then combined with the user's query to generate a response using the Google Gemini model.

### Text Loader
- TextLoader is used to extract text from the text file.
- The text is then split into smaller chunks with overlap using TokenTextSplitter.
- HuggingFaceEmbeddings is used to generate embeddings for the text chunks.
- FAISS is used to create a vector store from the document chunks.
- The vector store is then used to retrieve relevant document chunks based on the user's query.
- The retrieved document chunks are then combined with the user's query to generate a response using the Google Gemini model.

## Troubleshooting

- If you encounter any issues, please check the error message and try to resolve it.
- If you are still having issues, please contact the author for assistance.

## License

MIT License

## Author

Praveen Kumar
