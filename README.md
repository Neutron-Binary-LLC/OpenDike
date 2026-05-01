# OpenDike: Layered Morality Deducer

OpenDike is a lightweight morality wrapper and gateway designed to sit in front of downstream LLMs. It uses a hierarchical moral reasoning system to generate composite steering vectors based on cultural, community, demographic, and personal layers.

## Architecture & Workflow

### System Architecture
```mermaid
graph TD
    User([User Query]) --> Wrapper[Morality Wrapper]
    Wrapper --> Deducer[Layered Morality Deducer]
    
    subgraph MemPalace [MemPalace: Hierarchical Memory]
        CL[Country Layer]
        ML[Community Layer]
        DL[Demographic Layer]
        PL[Personal Layer]
    end
    
    Deducer <--> MemPalace
    Deducer --> Vector[Composite Moral Vector]
    Vector --> Conflict[Conflict Detection]
    Conflict --> FinalVector[Final Steering Prefix]
    
    FinalVector --> LLM[Downstream LLM]
    LLM --> Response([Aligned Response])
    
    Response -.-> Learner[Continual Learner]
    Learner -.-> PL
```

### Request Sequence
```mermaid
sequenceDiagram
    participant U as User
    participant W as Morality Wrapper
    participant D as Deducer
    participant M as MemPalace
    participant L as LLM
    participant C as Continual Learner

    U->>W: Send Query
    W->>D: Request Moral Steering
    D->>M: Retrieve Traces (Walking the Palace)
    M-->>D: Layered Contextual Traces
    D->>D: Compose Moral Vectors & Detect Conflicts
    D-->>W: Composite Steering Vector
    W->>L: Query + Moral Prefix
    L-->>W: Aligned Response
    W-->>U: Final Response
    W->>C: Extract Salient Episode
    C->>M: Update Personal Layer
```

## Project Structure

- `src/opendike/`: Core logic and models.
    - `core.py`: Implementation of `MoralVector`, `MemPalace`, and `LayeredMoralityDeducer`.
    - `train.py`: Training stubs for future improvements (Transformer-based deduction).
- `data/`: Local storage for `MemPalace` traces and profiles.
- `tests/`: Unit and integration tests.
- `requirements.txt`: Python dependencies.
- `Dockerfile` & `docker-compose.yml`: Containerized runtime and training setup.

## Setup & Usage

### Local Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the core example:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src
   python src/opendike/core.py
   ```

### Docker Setup

Run the runtime service:
```bash
docker-compose up runtime
```

Run the training service (stub):
```bash
docker-compose up training
```

## Core Features

- **Hierarchical Reasoning**: Composes moral priorities from Country, Community, Demographic, and Personal layers.
- **MemPalace Memory**: Spatial/hierarchical storage for morally salient interaction traces.
- **Continual Learning**: Updates the personal layer based on user feedback and moral episodes.
- **Conflict Detection**: Identifies and reports moral disagreements between different layers.

## Real World Use Cases

- **Dynamic Parental Controls**: A "Teenager" demographic layer that emphasizes autonomy while a "Family" community layer maintains safety boundaries, evolving as the child grows.
- **Global Enterprise Compliance**: Regional layers ensure legal and cultural compliance (e.g., GDPR in EU, specific local customs in Asia) without needing separate model deployments.
- **Personalized Ethical AI**: Users can "train" their own local personal layer to reflect their specific values (e.g., veganism, specific religious views) which modulate the LLM's tone and framing.
- **Conflict Resolution Platforms**: Using the Conflict Detection feature to highlight where cultural norms differ from individual preferences, facilitating transparent dialogue.

## Suggested Features & Roadmap

### Short Term (v0.2 - v0.5)
- **Production Vector DB Integration**: Transition from simple dict-based storage to `ChromaDB` or `FAISS` for high-performance retrieval.
- **Expanded MFT Dimensions**: Add more granular dimensions like *Liberty/Oppression* and *Wastefulness/Efficiency*.
- **API Gateway Interface**: Wrap the Deducer in a FastAPI service for seamless integration with existing LLM pipelines.

### Mid Term (v0.6 - v1.0)
- **Nuanced Moral Deduction (Transformer)**: Replace heuristic trace-influence with a small fine-tuned transformer (e.g., a 125M parameter model) to better capture the relationship between context and moral weights.
- **LoRA Adapter Support**: Implement Low-Rank Adaptation (LoRA) for efficient per-layer updates without full model fine-tuning.
- **Transparency Dashboard**: A UI to visualize the "Palace Walk" and how different layers contributed to a specific response.

### Long Term (v1.5+)
- **Distributed/Federated Memory**: Implement a cloud/edge hybrid where broad layers (Country/Community) are updated via federated learning, while Personal layers remain strictly local for privacy.
- **Cross-Layer Reasoning**: Enable the Deducer to reason about *why* layers conflict and propose synthetic compromises.

## Scaling

- **Horizontal Scaling**: Deploy multiple Deducer instances behind a load balancer; broad layers are cached, personal layers are fetched per-session.
- **Privacy-First Storage**: Personal layers can be stored on-device (Edge AI) with only the resulting steering vectors sent to the cloud.
