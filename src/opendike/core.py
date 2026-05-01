from src.opendike import (
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
    
    # 2. Mock LLM function
    def mock_llm(prompt):
        return f"Mock Response to: {prompt[:100]}..."

    # 3. Scenario: Nordic Teenager
    context = {"country": "Nordic", "demographic": "teenager", "user_id": "user_456"}
    query = "Should I follow my parents' rules even if I disagree?"
    
    print("--- Initial Call ---")
    response = wrapper.call_llm(mock_llm, query, context)
    print(response)
    
    # 4. Continual Learning: User gives feedback
    feedback = "I liked that you prioritized my liberty, but you should respect authority more."
    episode = learner.extract_salient_episodes(query, response, feedback)
    if episode:
        learner.update_personal_layer("user_456", episode)
        print("\n[Learning] Personal layer updated with moral trace.")
    
    # 5. Subsequent call (should be influenced by trace)
    print("\n--- Subsequent Call (Should show increased Authority) ---")
    response_2 = wrapper.call_llm(mock_llm, query, context)
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
    print(wrapper.call_llm(mock_llm, conflict_query, conflict_context))

    # 7. Scenario: Organizational Context
    print("\n--- Organizational Scenario (TechCorp) ---")
    org_context = {
        "country": "US",
        "org_id": "TechCorp",
        "user_id": "dev_001"
    }
    org_query = "How should I handle a disagreement with a manager about project priority?"
    print(wrapper.call_llm(mock_llm, org_query, org_context))

if __name__ == "__main__":
    example_usage()
