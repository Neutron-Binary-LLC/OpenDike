import os
import sys
from opendike.config import config
from opendike import (
    MemPalace, 
    LayeredMoralityDeducer, 
    MoralityWrapper, 
    ContinualLearner
)

def run_interactive():
    print("=== OpenDike Interactive Test Script ===")
    
    # 1. Initialize components
    palace = MemPalace()
    deducer = LayeredMoralityDeducer(palace)
    wrapper = MoralityWrapper(deducer)
    learner = ContinualLearner(palace)
    
    # 2. LLM setup (Gemma, LM Studio, or Mock)
    use_gemma = config.get("wrapper.local_testing.use_gemma", False)
    use_lm_studio = config.get("wrapper.local_testing.use_lm_studio", False)
    
    if use_lm_studio:
        import requests
        lm_url = config.get("wrapper.local_testing.lm_studio_url", "http://127.0.0.1:1234/v1")
        print(f"Connecting to LM Studio at {lm_url}...")
        
        def lm_studio_llm(prompt):
            try:
                response = requests.post(
                    f"{lm_url}/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 1024
                    },
                    timeout=60
                )
                response.raise_for_status()
                resp_json = response.json()
                choice = resp_json["choices"][0]
                content = choice["message"].get("content")
                
                # If content is empty but reasoning_content exists, use that or a combination
                if not content and "reasoning_content" in choice["message"]:
                    content = choice["message"]["reasoning_content"]
                    
                return content or "No response content received from LM Studio."
            except Exception as e:
                return f"Error calling LM Studio: {e}"
        
        llm_func = lm_studio_llm
        print("LM Studio client initialized.")
    elif use_gemma:
        from transformers import pipeline
        import torch
        
        model_id = config.get("wrapper.local_testing.model_id", "google/gemma-2b")
        hf_token = os.environ.get("HF_TOKEN") or config.get("wrapper.local_testing.hf_token")
        
        print(f"Loading local Gemma model: {model_id}...")
        
        try:
            pipe = pipeline(
                "text-generation", 
                model=model_id, 
                device_map="auto",
                token=hf_token
            )
            
            def gemma_llm(prompt):
                outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
                return outputs[0]["generated_text"]
            
            llm_func = gemma_llm
            print("Gemma loaded successfully.")
        except Exception as e:
            print(f"Error loading Gemma: {e}")
            print("Falling back to Mock LLM.")
            def mock_llm(prompt):
                return f"[MOCK RESPONSE] {prompt[:100]}..."
            llm_func = mock_llm
    else:
        print("Using Mock LLM (Gemma disabled in config).")
        def mock_llm(prompt):
            return f"[MOCK RESPONSE] {prompt[:100]}..."
        llm_func = mock_llm

    user_id = "interactive_user_001"
    
    while True:
        print("\n" + "="*50)
        query = input("\nEnter your query (or 'exit' to quit): ").strip()
        if query.lower() in ['exit', 'quit']:
            break
        
        if not query:
            continue

        print("\nEnter Context (leave blank for defaults):")
        country = input("Country (e.g., Nordic, EastAsian, US): ").strip() or "US"
        community = input("Community (e.g., UrbanProgressive, RuralTraditional): ").strip() or "UrbanProgressive"
        demographic = input("Demographic (e.g., teenager, elder): ").strip() or "teenager"
        org_id = input("Organization ID (e.g., TechCorp, NonProfit): ").strip()
        
        context = {
            "user_id": user_id,
            "country": country,
            "community": community,
            "demographic": demographic
        }
        if org_id:
            context["org_id"] = org_id

        print("\n--- Processing ---")
        try:
            # Show deduction details first
            deduction = deducer.deduce_moral_vector(query, context)
            print(f"Conflicts detected: {deduction['conflicts']}")
            print(f"Reasoning: {deduction['reasoning']}")
            
            # Call LLM through wrapper
            response = wrapper.call_llm(llm_func, query, context)
            print("\n--- Aligned Response ---")
            print(response)
            
            # Feedback Loop
            print("\n--- Feedback ---")
            feedback = input("Provide feedback on the response (optional, press Enter to skip): ").strip()
            if feedback:
                # Update personal layer via learner (trace-based)
                episode = learner.extract_salient_episodes(query, response, feedback)
                if episode:
                    learner.update_personal_layer(user_id, episode)
                    print("[Learning] Personal layer trace stored.")
                
                # Update experts via deducer (adapter-based)
                user_feedback = {
                    "text": feedback,
                    "layer_id": user_id,
                    "satisfaction": 0.5 # default
                }
                deducer.update_from_feedback(query, deduction["composite_vector"], user_feedback, list(deduction["expert_outputs"].keys()))
                print("[Learning] Experts updated with feedback.")
                    
        except Exception as e:
            print(f"An error occurred: {e}")

    print("\nExiting interactive test. Goodbye!")

if __name__ == "__main__":
    # Ensure src is in path if not already
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    run_interactive()
