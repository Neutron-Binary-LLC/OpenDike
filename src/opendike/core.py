import os
from opendike.config import config
from opendike import (
    MemPalace, 
    LayeredMoralityDeducer, 
    MoralityWrapper, 
    ContinualLearner
)

def example_usage():
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
    elif use_gemma:
        from transformers import pipeline
        import torch
        
        model_id = config.get("wrapper.local_testing.model_id", "google/gemma-2b")
        hf_token = os.environ.get("HF_TOKEN") or config.get("wrapper.local_testing.hf_token")
        
        print(f"Loading local Gemma model: {model_id}...")
        
        try:
            # Note: Gemma is a gated model. Ensure you have access and are authenticated.
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
        except Exception as e:
            print(f"Error loading Gemma: {e}")
            print("Falling back to Mock LLM.")
            def mock_llm(prompt):
                return f"Mock Response to: {prompt[:100]}..."
            llm_func = mock_llm
    else:
        def mock_llm(prompt):
            return f"Mock Response to: {prompt[:100]}..."
        llm_func = mock_llm

    # 3. Scenario: Nordic Teenager
    context = {"country": "Nordic", "demographic": "teenager", "user_id": "user_456"}
    query = "Should I follow my parents' rules even if I disagree?"
    
    print("--- Initial Call ---")
    response = wrapper.call_llm(llm_func, query, context)
    print(response)
    
    # 4. Continual Learning: User gives feedback
    feedback = "I liked that you prioritized my liberty, but you should respect authority more."
    episode = learner.extract_salient_episodes(query, response, feedback)
    if episode:
        learner.update_personal_layer("user_456", episode)
        print("\n[Learning] Personal layer updated with moral trace.")
    
    # 5. Subsequent call (should be influenced by trace)
    print("\n--- Subsequent Call (Should show increased Authority) ---")
    response_2 = wrapper.call_llm(llm_func, query, context)
    print(response_2)
    
    # 6. Scenario: Cultural Conflict
    print("\n--- Conflict Scenario (Nordic vs Traditional) ---")
    conflict_context = {
        "country": "Nordic",
        "community": "RuralTraditional",
        "user_id": "user_789"
    }
    conflict_query = "Is it okay to publicly challenge a community elder?"
    deduction = deducer.deduce(conflict_query, conflict_context)
    print(f"Conflicts detected: {deduction['conflicts']}")
    print(wrapper.call_llm(llm_func, conflict_query, conflict_context))
    
    # 7. Scenario: Organizational Context
    print("\n--- Organizational Scenario (TechCorp) ---")
    org_context = {
        "country": "US",
        "org_id": "TechCorp",
        "user_id": "dev_001"
    }
    org_query = "How should I handle a disagreement with a manager about project priority?"
    print(wrapper.call_llm(llm_func, org_query, org_context))

if __name__ == "__main__":
    example_usage()
