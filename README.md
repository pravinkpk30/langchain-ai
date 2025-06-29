# LangChain AI Projects

This repository contains multiple LangChain-based projects demonstrating different AI capabilities and integrations. Each folder contains a self-contained project with its own documentation and setup instructions.

## Project Structure

### 1. `/api` - LangChain API Server
A FastAPI-based server that exposes LangChain functionality through REST endpoints. This includes:
- Essay generation
- Poem generation
- Integration with Google's Gemini model
- LangServe for serving LangChain models

### 2. `/rag` - Retrieval-Augmented Generation Examples
Multiple RAG (Retrieval-Augmented Generation) implementations for different document types:
- **PDF Document Processing**: Extract and process text from PDF files
- **Web Content Processing**: Scrape and process content from web pages
- **Text File Processing**: Load and process plain text documents
- Vector similarity search using FAISS and Chroma

### 3. `/retriever-chain` - Advanced Document Retrieval System
A more sophisticated document retrieval system featuring:
- PDF document processing with PyPDF
- Advanced text chunking and embedding
- Vector similarity search with FAISS
- Integration with Google's Gemini for question answering
- Streamlit-based web interface

### 4. `/chatbot` - Interactive Chatbot
A Streamlit-based chatbot interface with support for:
- Google Gemini integration
- Local Ollama model support
- Conversation history
- Simple and intuitive UI

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Google Gemini API key (for cloud-based features)
- Ollama (for local model support)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd langchain-ai

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

1. Navigate to the desired project directory:
   ```bash
   cd <project-directory>
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run <application-file>
   ```

3. Open your browser and navigate to `http://localhost:8501`

## Troubleshooting

- If you encounter any issues, please check the error message and try to resolve it.
- If you are still having issues, please contact the author for assistance.

## License

MIT License

## Author

Praveen Kumar
