
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
