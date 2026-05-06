import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from opendike import (
    MemPalace, 
    LayeredMoralityDeducer, 
    MoralityWrapper
)
from opendike.config import config
import requests

def test_full_flow_with_lm_studio():
    print("=== Testing Full Flow with LM Studio ===")
    
    # 1. Initialize
    palace = MemPalace()
    deducer = LayeredMoralityDeducer(palace)
    wrapper = MoralityWrapper(deducer)
    
    # 2. LLM Setup (API Client)
    lm_url = config.get("wrapper.local_testing.lm_studio_url", "http://127.0.0.1:1234/v1")
    
    def llm_func(prompt):
        print(f"\n--- Prompt sent to LM Studio ---\n{prompt[:200]}...\n")
        try:
            response = requests.post(
                f"{lm_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 256
                },
                timeout=30
            )
            response.raise_for_status()
            resp_json = response.json()
            choice = resp_json["choices"][0]
            content = choice["message"].get("content")
            
            # Handle models that return reasoning in a separate field
            if not content and "reasoning_content" in choice["message"]:
                content = choice["message"]["reasoning_content"]
                
            return content or "No response content received from LM Studio."
        except Exception as e:
            return f"Error calling LM Studio: {e}"

    # 3. Query
    context = {"country": "India_PenalCode", "community": "Christian_Bible"}
    query = "Is it acceptable to forgive a thief if they return the stolen goods?"
    
    print(f"Query: {query}")
    print(f"Context: {context}")
    
    try:
        response = wrapper.call_llm(llm_func, query, context)
        print("\n--- Final Response ---")
        print(response)
    except Exception as e:
        print(f"Flow failed: {e}")

    # 3. Query
    context = {"country": "India_PenalCode", "community": "Muslim_Koran"}
    query = "Is it acceptable to forgive a thief if they return the stolen goods?"

    print(f"Query: {query}")
    print(f"Context: {context}")

    try:
        response = wrapper.call_llm(llm_func, query, context)
        print("\n--- Final Response ---")
        print(response)
    except Exception as e:
        print(f"Flow failed: {e}")

if __name__ == "__main__":
    test_full_flow_with_lm_studio()
