import requests
import json
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from opendike.config import config

def test_lm_studio_connection():
    lm_url = config.get("wrapper.local_testing.lm_studio_url", "http://127.0.0.1:1234/v1")
    print(f"Testing connection to LM Studio at {lm_url}...")
    
    try:
        # Check models endpoint first to see if server is up
        resp = requests.get(f"{lm_url}/models", timeout=5)
        if resp.status_code == 200:
            print("Successfully connected to LM Studio server!")
            models = resp.json()
            print(f"Available models: {[m['id'] for m in models.get('data', [])]}")
            
            # Try a simple completion
            print("Attempting a simple chat completion...")
            chat_resp = requests.post(
                f"{lm_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Say 'Connection Successful'"}],
                    "temperature": 0.0,
                    "max_tokens": 10
                },
                timeout=10
            )
            if chat_resp.status_code == 200:
                result = chat_resp.json()["choices"][0]["message"]["content"]
                print(f"LM Studio Response: {result}")
            else:
                print(f"Chat completion failed with status: {chat_resp.status_code}")
        else:
            print(f"Failed to connect to /models. Status: {resp.status_code}")
    except Exception as e:
        print(f"Error: Could not connect to LM Studio at {lm_url}. Make sure LM Studio is running and the server is started.")
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_lm_studio_connection()
