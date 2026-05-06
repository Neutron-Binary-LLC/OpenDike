import os
import sys
from opendike.experts import LayeredMoralityDeducer
from opendike.memory import MemPalace

def test_expert_embeddings():
    print("Testing retrieval of expert embeddings from MemPalace...")
    palace = MemPalace()
    deducer = LayeredMoralityDeducer(palace)
    
    queries = [
        ("holy sacred tradition", "community", "Christian_Bible"),
        ("allah prophet mercy", "community", "Muslim_Koran"),
        ("law justice penal code", "country", "India_PenalCode")
    ]
    
    all_passed = True
    for query, layer_type, layer_id in queries:
        print(f"\nQuery: '{query}' for {layer_type}:{layer_id}")
        traces = palace.retrieve_traces(
            layer_type=layer_type,
            layer_id=layer_id,
            query=query,
            top_k=3,
            min_relevance=0.3
        )
        
        if traces:
            print(f"✓ Found {len(traces)} matching traces.")
            for i, trace in enumerate(traces):
                print(f"  [{i+1}] {trace.content[:100]}...")
        else:
            print(f"✗ No traces found for {layer_type}:{layer_id}.")
            all_passed = False
            
    if all_passed:
        print("\nExpert embeddings successfully verified in MemPalace!")
    else:
        print("\nSome expert embeddings failed to retrieve.")

if __name__ == "__main__":
    # Ensure src is in path
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
        
    test_expert_embeddings()
