from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import requests
import os

# Load environment variables
load_dotenv()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,  # Slightly higher temperature for more creative responses
    api_key=os.getenv("GOOGLE_API_KEY")
)


# Initialize Tools with better error handling
def get_tools():
    # Wikipedia Tool with better configuration
    wikipedia_api_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=200,
        lang="en",
        load_all_available_meta=False
    )
    wikipedia_tool = WikipediaQueryRun(
        api_wrapper=wikipedia_api_wrapper,
        name="wikipedia",
        description="Useful for general knowledge questions and factual information."
    )
    
    # Arxiv Tool with better configuration
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=2,  # Get 2 results for better context
        doc_content_chars_max=500,  # Increased character limit
        load_max_docs=2,
        load_all_available_meta=True
    )
    arxiv_tool = ArxivQueryRun(
        api_wrapper=arxiv_wrapper,
        name="arxiv",
        description="Useful for finding and summarizing academic papers and research."
    )
    
    return [wikipedia_tool, arxiv_tool]

# Get tools
tools = get_tools()

# Add error handling for tool execution
# for tool in tools:
#     original_run = tool.run
#     def safe_run(*args, **kwargs):
#         try:
#             result = original_run(*args, **kwargs)
#             if not result or result.strip() == "":
#                 return "No relevant information found."
#             return result
#         except Exception as e:
#             return f"Error accessing {tool.name}: {str(e)}"
#     tool.run = safe_run

# Convert tools to OpenAI function format
functions = [format_tool_to_openai_function(tool) for tool in tools]

# Enhanced prompt with better instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Use the available tools to find accurate information.
    - Be concise but informative
    - If a tool fails, try another approach
    - If no information is found, say so honestly
    - Always cite your sources when possible"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent with better error handling
agent = {
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_functions(x.get('intermediate_steps', [])),
    "chat_history": lambda x: x.get("chat_history", [])
} | prompt | llm.bind(functions=functions) | OpenAIFunctionsAgentOutputParser()

# Create the agent executor with better error handling
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    max_iterations=5,  # Limit to prevent infinite loops
    early_stopping_method="generate"  # Better handling of tool failures
)

# Wrapper function to handle tool execution with error handling
def safe_invoke(executor, input_data):
    try:
        response = executor.invoke(input_data)
        output = response.get('output', 'I apologize, but I encountered an error processing your request.')
        
        # Check if the output indicates an error
        if "error" in output.lower() or "timed out" in output.lower():
            return "I'm having trouble accessing that information right now. Please try again later or ask a different question."
        return output
    except Exception as e:
        return f"I'm sorry, but I encountered an error: {str(e)}"


# Enhanced main execution with better error handling
if __name__ == "__main__":
    chat_history = []
    print("Welcome! Ask me anything or type 'exit' to quit.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if not user_input:
                print("Please enter a valid question.")
                continue
                
            print("\nThinking...")
            
            try:
                # Get the response using our safe invoke wrapper
                answer = safe_invoke(agent_executor, {
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                print(f"\nAssistant: {answer}")
                
                # Update chat history
                chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=answer)
                ])
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                print(f"\nAssistant: I'm sorry, but I encountered an error: {error_msg}")
                print("Please try rephrasing your question or try again later.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")