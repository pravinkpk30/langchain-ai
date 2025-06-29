# LangChain Chatbot with LangSmith Monitoring

A simple yet powerful chatbot built with LangChain and Google's Gemini model, featuring monitoring capabilities with LangSmith.

## Features

- Interactive chatbot interface using Streamlit
- Powered by Google's Gemini 2.0 Flash model
- Integrated with LangSmith for monitoring and observability
- Easy to set up and customize

## Prerequisites

- Python 3.8+
- Google API key with access to Gemini models
- LangSmith account and API key (for monitoring)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd langchain-ai
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=your_project_name
   ```

## Running the Application

1. Navigate to the chatbot directory:
   ```bash
   cd chatbot
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

## Monitoring with LangSmith

This project is integrated with LangSmith for monitoring and observability. To use LangSmith:

1. Sign up for a LangSmith account at [https://smith.langchain.com](https://smith.langchain.com)
2. Create an API key in your account settings
3. Add the following environment variables to your `.env` file:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=your_project_name
   ```

### Key Benefits of LangSmith Monitoring:

- **Tracing**: Visualize the execution flow of your LangChain applications
- **Debugging**: Identify issues in your prompts and chains
- **Performance Monitoring**: Track latency and token usage
- **Quality Evaluation**: Monitor the quality of model outputs
- **Collaboration**: Share and review traces with your team

## Project Structure

```
langchain-ai/
├── chatbot/
│   └── app.py          # Main Streamlit application
├── .env                # Environment variables
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Customization

### Changing the Model

You can modify the model in `chatbot/app.py` by changing the model name:

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
```

### Modifying the Prompt

Edit the prompt template in `chatbot/app.py` to change the bot's behavior:

```python
prompt_template = """
You are a helpful assistant that can answer questions about the world.
Question: {question}
Answer: """
```

## Local Ollama Setup

This project supports running with a local Ollama model instead of using Google's Gemini API. Here's how to set it up:

### Prerequisites

1. Install [Ollama](https://ollama.ai/) on your local machine
2. Pull the desired model (e.g., llama3.2):
   ```bash
   ollama pull llama3.2
   ```

### Running with Ollama

1. Navigate to the chatbot directory:
   ```bash
   cd chatbot
   ```

2. Run the Streamlit application with Ollama:
   ```bash
   streamlit run localollama.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

### Key Differences from Gemini Version

- No API key required
- Runs entirely locally on your machine
- Uses the Ollama framework to manage local LLMs
- May have different performance characteristics compared to cloud-based models

### Customizing the Ollama Model

You can modify the model in `chatbot/localollama.py` by changing the model name:

```python
# Change "llama3.2" to any model you've downloaded with Ollama
llm = Ollama(model="llama3.2")
```

### Troubleshooting Ollama

- Ensure Ollama is running in the background
- Verify the model is downloaded using `ollama list`
- Check available models at [Ollama Library](https://ollama.ai/library)
- For GPU acceleration, ensure you have the correct drivers installed

## License

MIT License

## Troubleshooting

- If you encounter API key errors, ensure your `.env` file is properly set up
- For LangSmith issues, verify your API key and project name in the environment variables
- Check the Streamlit logs in the terminal for any runtime errors

## License

This project is licensed under the MIT License - see the LICENSE file for details.
