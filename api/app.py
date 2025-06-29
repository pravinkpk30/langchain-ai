from fastapi import FastAPI 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    api_key=os.getenv("GOOGLE_API_KEY"), 
    temperature=0.7
)

# Define input schema
class TopicInput(BaseModel):
    topic: str
    config: Dict[str, Any] = {}

# Initialize FastAPI app
app = FastAPI(
    title="LangChain API",
    description="LangChain API for generating essays and poems using Google's Gemini",
    version="0.1.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
)

# Define prompts
essay_prompt = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with 100 words"
)

poem_prompt = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5-year-old child with 100 words. "
    "Use simple language and make it fun and engaging for young children."
)

# Add routes with input schemas
add_routes(
    app,
    essay_prompt | llm,
    path="/essay"
)

add_routes(
    app,
    poem_prompt | llm,
    path="/poem"
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the LangChain API. Visit /docs for the API documentation."}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
