import os
import sys
import numpy as np
from opendike.experts import LayeredMoralityDeducer
from opendike.memory import MemPalace

def test_lora_adaptation():
    print("Testing LoRA adaptation loading and effect...")
    palace = MemPalace()
    deducer = LayeredMoralityDeducer(palace)
    
    # Check specifically for experts with adapters
    experts_to_test = [
        "community:Christian_Bible",
        "community:Muslim_Koran",
        "country:India_PenalCode"
    ]
    
    for exp_key in experts_to_test:
        if exp_key in deducer.experts:
            expert = deducer.experts[exp_key]
            print(f"\n--- Expert: {exp_key} ---")
            print(f"Base Profile: {expert.base_profile.to_numpy()}")
            print(f"Adapter: {expert.adapter}")
            
            # Get vector with no traces/feedback
            result_mv = expert.get_vector("test query")
            result_vec = result_mv.to_numpy()
            print(f"Final Vector: {result_vec}")
            
            # Verify if adapter was applied
            if expert.adapter and "bias" in expert.adapter:
                bias = np.array(expert.adapter["bias"])
                expected = np.clip(expert.base_profile.to_numpy() + bias, 0.0, 1.0)
                if np.allclose(result_vec, expected):
                    print("✓ LoRA Adapter bias correctly applied to final vector.")
                else:
                    print("✗ LoRA Adapter bias NOT correctly applied.")
            else:
                print("! No adapter bias found for this expert.")
        else:
            print(f"✗ Expert {exp_key} NOT found.")

if __name__ == "__main__":
    # Ensure src is in path
    current_dir = os.getcwd()
    src_dir = os.path.join(current_dir, "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
        
    test_lora_adaptation()
