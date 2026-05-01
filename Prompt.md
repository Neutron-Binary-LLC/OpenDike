You are an expert AI systems architect and Python engineer specializing in LLM alignment, guardrails, hierarchical systems, and continual learning. Produce clean, production-ready, modular code with explanations.
User Prompt:
You are tasked with generating a complete, runnable Python implementation for a Layered Morality Deducer — a lightweight morality wrapper / gateway that sits in front of any downstream LLM.
Core Requirements:
Architecture Overview:

It uses a hierarchical (layered) moral reasoning system:
Base Layer → Country / Region
Intermediate Layers → Community / Sub-culture, Demographic (age band e.g. "teenager", gender)
Top Layer → Individual / Session-level personalization
It maintains a Mem-Palace style distributed and layered memory for storing only morally relevant traces from past interactions.
It performs continual deep learning (lightweight updates) on those traces to upgrade weights/biases or embeddings per layer.
For every new query + context, it outputs a composite moral steering vector (or structured JSON) that can be prepended as a system prompt or used for activation steering / post-filtering on the downstream LLM.

Key Components to Implement:

Moral Vector Schema
Use an extended Moral Foundations Theory (MFT) + additional dimensions.
Example dimensions: Care/Harm, Fairness/Proportionality, Loyalty, Authority, Sanctity/Purity, Liberty, plus any others you deem useful.
Each layer produces a vector (dict or numpy array) of weights (0.0–1.0) + optional constraints and reasoning style hints.

MemPalace-inspired Hierarchical Storage
Implement a spatial/hierarchical memory structure (Wings → Halls/Rooms → Drawers) where:
Broad layers (country/community) are shared or semi-shared.
Personal layer is user-specific.

Only store morally relevant traces (user feedback, value conflicts, explicit moral statements, resolved dilemmas). Ignore pure chit-chat.
Use embeddings (e.g., via sentence-transformers or lightweight model) + vector DB (Chroma, FAISS, or simple dict for prototype) for retrieval.
Support "walking the palace" to retrieve relevant traces per layer.

Layered Morality Deducer
A small, fast model or rule+embedding hybrid that:
Takes: query summary (non-PII), layer identifiers (country, community, age_band, etc.), retrieved traces.
Outputs: per-layer moral vectors.

Compose layers into a final steering vector (e.g., weighted sum or attention-style fusion, with conflict detection).

Continual Learning Loop
After each interaction, extract morally salient episodes.
Perform lightweight updates: LoRA-style adapters, embedding updates, or simple gradient steps on a small scorer model (or use replay buffer + fine-tuning on a distilled model).
Prevent catastrophic forgetting with regularization or periodic replay of core moral scenarios.

Output to Downstream LLM
Generate a concise system prompt prefix or JSON steering object that includes:
Moral priority weights
Forbidden/encouraged framings
Required considerations per layer
Transparency note (e.g., "Country layer emphasizes Authority 0.75; Personal teen layer boosts Care 0.85")



Technical Constraints:

Keep the deducer low-latency (<150-200ms added overhead ideally).
Support non-PII inputs only (coarse demographic flags provided by user or session).
Use modern 2026-friendly libraries: pydantic, sentence-transformers, chromadb or faiss, torch (with quantization), langchain or plain Python where possible.
Make it modular and extensible (easy to plug in different LLMs as the downstream model).
Include safety floors: hard global constraints against direct harm/incitement that override layers.

Deliverables:

Full working prototype code structure (main classes: MoralVector, MemPalace, LayeredMoralityDeducer, MoralityWrapper).
Example usage: how to wrap a call to any LLM (e.g., Anthropic, OpenAI, local vLLM).
Sample data for initialization (synthetic country/community profiles based on MFT).
Clear comments explaining how layers compose and how continual learning happens.
Suggestions for future improvements (e.g., replacing hybrid with a small fine-tuned transformer).

Generate clean, well-documented code with type hints. Start with a minimal viable version that can run locally, then note where to scale it.
Focus on pluralism: the system should surface conflicts between layers rather than silently forcing one moral view.
Begin your response with the code.