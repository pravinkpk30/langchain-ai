# Groq RAG Chat Application

A Streamlit-based chat application that demonstrates Retrieval-Augmented Generation (RAG) using Groq's LLM and LangChain.

## Overview

This application implements a RAG (Retrieval-Augmented Generation) pipeline that:
- Loads and processes web content (default: LangChain documentation)
- Creates a vector store for efficient similarity search
- Provides a chat interface to ask questions
- Generates answers using Groq's Llama 3 8B model
- Shows the source documents used for generating answers

## Features

- **Document Processing**: Automatically loads and chunks web content
- **Vector Search**: Uses FAISS for efficient similarity search
- **Chat Interface**: Simple and intuitive web interface
- **Source Tracking**: Shows which document chunks were used for each answer
- **Performance Metrics**: Displays timing information for retrieval and generation

## Prerequisites

- Python 3.8+
- Groq API key (sign up at [Groq Cloud](https://groq.com/))
- Required Python packages (see Installation)

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```
   Additional requirements:
   ```bash
   pip install streamlit langchain-groq faiss-cpu sentence-transformers python-dotenv
   ```

3. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open the provided URL in your web browser

3. Enter your question in the text input and press Enter

4. View the generated answer and expand the "Document Similarity Search" section to see the source chunks used

## Configuration

You can customize the application by modifying these parameters in `app.py`:

- `WebBaseLoader` URL: Change the default documentation source
- `chunk_size` and `chunk_overlap` in `RecursiveCharacterTextSplitter`
- `model_name`: Change the Groq model (default: "llama3-8b-8192")
- `temperature`: Adjust the creativity of responses (default: 0.2)

## How It Works

1. **Document Loading**: The app loads content from the specified URL using WebBaseLoader
2. **Text Processing**: Documents are split into chunks with overlapping text
3. **Embedding Generation**: Text chunks are converted to vector embeddings using HuggingFace's sentence transformers
4. **Vector Storage**: Embeddings are stored in a FAISS index for fast similarity search
5. **Query Processing**: User questions are converted to embeddings and used to find relevant document chunks
6. **Response Generation**: The most relevant chunks are passed to the LLM to generate an answer

## Troubleshooting

- **API Key Issues**: Ensure your `.env` file is in the correct location and contains a valid Groq API key
- **Dependency Errors**: Make sure all required packages are installed with the correct versions
- **Memory Issues**: For large documents, you might need to reduce the number of documents processed or chunk size

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- [LangChain](https://www.langchain.com/) for the RAG framework
- [Groq](https://groq.com/) for the LLM API
- [HuggingFace](https://huggingface.co/) for the sentence transformers
- [Streamlit](https://streamlit.io/) for the web interface
