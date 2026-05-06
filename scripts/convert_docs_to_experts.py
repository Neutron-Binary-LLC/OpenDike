import os
import json
import yaml
import numpy as np
import sys
from typing import Dict, List, Any

# Ensure src is in path for opendike imports
current_dir = os.getcwd()
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from opendike.memory import MemPalace

# Mocking PDF extraction if libraries are missing, or using a simple strategy
# In a real scenario, we'd use pypdf or pdfplumber
try:
    from pypdf import PdfReader
    HAS_PDF_LIB = True
except ImportError:
    HAS_PDF_LIB = False
    print("Warning: pypdf not installed. Using placeholder text for analysis.")

# MFT Keywords for simple heuristic-based moral vector deduction
MFT_KEYWORDS = {
    "care_harm": ["care", "harm", "help", "suffer", "protect", "hurt", "kindness", "compassion", "safety"],
    "fairness_proportionality": ["fair", "unfair", "justice", "equality", "rights", "proportional", "cheat", "bias", "honest"],
    "loyalty_betrayal": ["loyal", "betray", "group", "country", "family", "nation", "community", "together", "traitor", "fidelity"],
    "authority_subversion": ["authority", "obey", "respect", "tradition", "elder", "order", "leader", "rank", "hierarchy", "subvert"],
    "sanctity_degradation": ["sacred", "holy", "purity", "degrade", "sin", "clean", "disgust", "god", "divine", "nature"],
    "liberty_oppression": ["liberty", "freedom", "oppress", "bully", "choice", "rights", "autonomy", "coercion", "independent"]
}

def extract_text_from_pdf(pdf_path: str) -> str:
    if not HAS_PDF_LIB:
        # Return a simulated extraction or generic text based on filename
        filename = os.path.basename(pdf_path).lower()
        if "bible" in filename:
            return "holy sacred god love neighbor obey commandments tradition purity"
        elif "koran" in filename:
            return "allah mercy prayer holy sacred authority prophet tradition purity"
        elif "penal" in filename:
            return "law penal code justice authority punishment rights crime legal order"
        return ""
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        # Read only first 100 pages to avoid OOM or slow processing for very large PDFs
        for i in range(min(len(reader.pages), 100)):
            text += reader.pages[i].extract_text() + " "
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def estimate_moral_vector(text: str) -> Dict[str, Any]:
    text = text.lower()
    scores = {}
    for foundation, keywords in MFT_KEYWORDS.items():
        count = sum(text.count(word) for word in keywords)
        scores[foundation] = count
    
    # Normalize to 0.0 - 1.0 range
    max_score = max(scores.values()) if scores.values() and max(scores.values()) > 0 else 1
    normalized = {k: min(1.0, 0.3 + (v / max_score) * 0.7) for k, v in scores.items()}
    
    # Add deontological_vs_utilitarian heuristic
    # Religious texts often more deontological
    deontological = 0.5
    if any(word in text for word in ["commandment", "forbidden", "holy", "sacred", "law", "shall"]):
        deontological = 0.8
        
    normalized["deontological_vs_utilitarian"] = deontological
    normalized["reasoning"] = "Extracted from document via MFT keyword analysis."
    
    return normalized

def train_lora_adaptation(text: str, base_vector: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate LoRA adaptation by processing text in chunks and refining a bias/adapter.
    """
    print("  Training LoRA adaptation...")
    # Split text into chunks (e.g., paragraphs or sentences)
    chunks = [c.strip() for c in text.split(".") if len(c.strip()) > 20]
    if not chunks:
        return {"bias": [0.0] * 7}

    # Initialize bias
    bias = np.zeros(7)
    learning_rate = 0.05
    
    # Map foundations to indices for convenience
    foundation_map = {
        "care_harm": 0,
        "fairness_proportionality": 1,
        "loyalty_betrayal": 2,
        "authority_subversion": 3,
        "sanctity_degradation": 4,
        "liberty_oppression": 5,
        "deontological_vs_utilitarian": 6
    }

    # Process chunks to find local moral intensity
    for chunk in chunks:
        chunk = chunk.lower()
        chunk_scores = {}
        for foundation, keywords in MFT_KEYWORDS.items():
            count = sum(chunk.count(word) for word in keywords)
            if count > 0:
                idx = foundation_map[foundation]
                # If a chunk strongly mentions a foundation, we nudge the bias
                # This simulates finding "salient episodes" in the document
                intensity = min(1.0, count / 5.0) 
                # Nudge towards higher intensity if found
                target = 1.0 if intensity > 0.5 else 0.0
                current_val = base_vector.get(foundation, 0.5) + bias[idx]
                delta = (target - current_val) * learning_rate * intensity
                bias[idx] += delta

    # Clip bias to reasonable range (e.g., -0.2 to 0.2) to keep it "LoRA-like" (fine-tuning)
    bias = np.clip(bias, -0.2, 0.2)
    
    return {"bias": bias.tolist()}

def convert_docs_to_experts(docs_dir: str, config_path: str):
    pdfs = [
        "Community_Christian_Bible.pdf",
        "Community_Muslim_Koran.pdf",
        "Country_India_PenalCode.PDF"
    ]
    
    new_experts = {}
    
    # Initialize MemPalace for embedding generation and storage
    palace = MemPalace()
    
    for pdf_name in pdfs:
        full_path = os.path.join(docs_dir, pdf_name)
        if not os.path.exists(full_path):
            print(f"Skipping {pdf_name}, file not found.")
            continue
            
        print(f"Processing {pdf_name}...")
        text = extract_text_from_pdf(full_path)
        
        # 1. Base vector via keyword summary
        vector = estimate_moral_vector(text)
        
        # 2. LoRA adaptation via chunk processing
        adapter = train_lora_adaptation(text, vector)
        vector["adapter"] = adapter
        
        # Naming convention: LayerType_LayerID_
        # Parse from filename e.g. Community_Christian_Bible.pdf -> Community_Christian_Bible_
        name_parts = os.path.splitext(pdf_name)[0].split("_")
        # Ensure it follows LayerType_LayerID_
        if len(name_parts) >= 2:
            layer_type = name_parts[0].lower() # Normalize to match existing lowercase keys
            layer_id = "_".join(name_parts[1:])
            expert_name = f"{layer_type}_{layer_id}_"
        else:
            layer_type = "unknown"
            layer_id = name_parts[0]
            expert_name = f"unknown_{layer_id}_"
            
        if layer_type not in new_experts:
            new_experts[layer_type] = {}
        
        new_experts[layer_type][layer_id] = vector
        print(f"Created expert: {expert_name}")

        # 3. Create embeddings for document chunks and store in MemPalace
        print(f"  Generating embeddings for {pdf_name} chunks...")
        chunks = [c.strip() for c in text.split(".") if len(c.strip()) > 50]
        for chunk in chunks:
            palace.add_trace(
                layer_type=layer_type,
                layer_id=layer_id,
                content=chunk,
                metadata={"source": pdf_name}
            )

    # Save memory storage
    palace.save_storage()
    print("Saved expert embeddings to MemPalace storage.")

    # Update config.yaml
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
            
        if "deducer" not in config_data:
            config_data["deducer"] = {}
        if "base_profiles" not in config_data["deducer"]:
            config_data["deducer"]["base_profiles"] = {}
            
        for l_type, profiles in new_experts.items():
            if l_type not in config_data["deducer"]["base_profiles"]:
                config_data["deducer"]["base_profiles"][l_type] = {}
            config_data["deducer"]["base_profiles"][l_type].update(profiles)
            
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False)
        print(f"Updated {config_path} with new experts.")
    else:
        print(f"Config file {config_path} not found. Saving to new_experts.json")
        with open("new_experts.json", "w") as f:
            json.dump(new_experts, f, indent=2)

if __name__ == "__main__":
    convert_docs_to_experts("docs", "config.yaml")
