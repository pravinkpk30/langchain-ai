from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from langchain_community.llms import Ollama

load_dotenv()

# Ollama llm with model llama3.2 and run locally
llm = Ollama(model="llama3.2")

# Create a prompt template
prompt_template = """
You are a helpful assistant that can answer questions about the world.
Question: {question}
Answer: """

prompt = ChatPromptTemplate.from_template(prompt_template)

# streamlit ui
st.title("LangChain Chatbot")

question = st.text_input("Search the topic you want to know about")

if question:
    with st.spinner("Thinking..."):
        chain = prompt | llm | StrOutputParser()    # chain the prompt, llm and output parser
        response = chain.invoke({"question": question})
        st.write(response)

