import os
import json
import logging
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def get_llm_client():
    """Get an instance of the LLM client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    return ChatGroq(
        temperature=0.7,
        groq_api_key=api_key,
        model_name="mixtral-8x7b-32768"
    )

async def run_llm_call(prompt: str) -> str:
    """Run a single LLM call with the given prompt"""
    try:
        llm = get_llm_client()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logging.error(f"Error during LLM call: {str(e)}")
        return ""
