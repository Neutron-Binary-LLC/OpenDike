import os
import sys
from opendike.experts import LayeredMoralityDeducer
from opendike.memory import MemPalace

def test_new_experts():
    print("Testing loading of new experts...")
    palace = MemPalace()
    deducer = LayeredMoralityDeducer(palace)
    
    
    
    # Check if experts are registered
    expected_experts = [
        "community:Christian_Bible",
        "community:Muslim_Koran",
        "country:India_PenalCode"
    ]
    
    all_registered = True
    for exp_key in expected_experts:
        if exp_key in deducer.experts:
            print(f"✓ Expert {exp_key} is registered.")
            expert = deducer.experts[exp_key]
            print(f"  Reasoning: {expert.base_profile.reasoning}")
        else:
            print(f"✗ Expert {exp_key} is NOT registered.")
            all_registered = False
            
    if all_registered:
        print("\nAll new experts successfully loaded into OpenDike!")
    else:
        print("\nSome experts failed to load.")

if __name__ == "__main__":
    # Ensure src is in path
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
        
    test_new_experts()
