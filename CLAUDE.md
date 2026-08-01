# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the interactive CLI (primary entry point):**
```bash
uv run opendike-run
```

**Run the training simulation:**
```bash
uv run opendike-train
```

**Run the example flow (with LLM backend configured):**
```bash
uv run python src/opendike/core.py
```

**Install editable:**
```bash
uv pip install -e .
```

**Without uv (set PYTHONPATH first):**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/opendike/core.py
```

**Run a script directly:**
```bash
uv run python scripts/<script_name>.py
```

There is no test suite yet — `scripts/` contains ad-hoc verification scripts (e.g., `verify_full_flow.py`, `test_expert_embeddings.py`).

## Architecture

OpenDike is a **moral alignment gateway** that sits in front of any downstream LLM. Given a user query and context, it produces a composite moral steering vector that is prepended as a system prefix to steer the LLM response.

### Request flow

```
User Query + Context
  → MoralityWrapper.call_llm()
      → LayeredMoralityDeducer.deduce_moral_vector()
          → MoralGatingNetwork.route()         # compute per-layer weights
          → MoralExpert.get_vector()           # one per active layer
              → MemPalace.retrieve_traces()    # cosine-sim lookup of past traces
          → MoralVector.weighted_average()     # fuse expert outputs
          → conflict detection (dim-wise variance > 0.4)
      → _generate_system_prefix()             # convert vector → text prefix
  → llm_func(prefix + query)
  → aligned response
```

### Core components

| Module | Class | Role |
|---|---|---|
| `models.py` | `MoralVector` | 7-dimension float vector (MFT-based). Serializes to/from numpy, dict. |
| `models.py` | `MoralTrace` | A stored interaction — content + embedding + metadata (e.g. `vector_delta`). |
| `memory.py` | `MemPalace` | Hierarchical in-memory + disk store: `{layer_type → layer_id → [MoralTrace]}`. Uses `sentence-transformers` for cosine-similarity retrieval. Persists to `data/memory_storage.json`. |
| `experts.py` | `MoralExpert` | One expert per `layer_type:layer_id` pair. Retrieves traces from MemPalace, applies age-decayed deltas, and optionally applies a bias adapter. |
| `experts.py` | `MoralGatingNetwork` | Computes routing weights. Supports `layered` mode (additive intent boosts) and `expert` mode (aggressive redistribution). Intent keywords are config-driven. |
| `experts.py` | `LayeredMoralityDeducer` | Orchestrates experts and gating. The main entry point is `deduce_moral_vector(query, context)`. Loads expert `base_profile`s from `config.yaml` at init. |
| `wrapper.py` | `MoralityWrapper` | Calls `deduce_moral_vector`, formats the result as a text prefix, then calls the supplied `llm_func(prompt)`. |
| `learning.py` | `ContinualLearner` | Extracts morally salient episodes from feedback and stores traces in MemPalace to influence future personal-layer responses. |
| `train.py` | `ContinuousTrainer` | Recalibrates expert `base_profile`s by computing mean `vector_delta` from stored traces ("centroid" method). Also houses `GatingNetwork` (PyTorch) for future trainable routing. |
| `config.py` | `Config` | Singleton that loads `config.yaml` (discovered from CWD or project root, or via `OPENDIKE_CONFIG` env var). Accessed via dot-path: `config.get("experts.default_weights")`. |

### MoralVector dimensions (index order matters for numpy ops)

```
0: care_harm
1: fairness_proportionality
2: loyalty_betrayal
3: authority_subversion
4: sanctity_degradation
5: liberty_oppression
6: deontological_vs_utilitarian   (0 = utilitarian, 1 = deontological)
```

### Context keys → layer mapping

When calling `deduce_moral_vector(query, context)`, context keys are mapped to layer types:
- `user_id` → `personal`
- `org_id` → `organization`
- Any other key (e.g. `country`, `community`, `demographic`) → used directly as the layer type

Expert lookup key is `"{layer_type}:{layer_id}"`. If not found, falls back to `"{layer_type}:default"`.

## Configuration

All tunable behaviour lives in `config.yaml`. Key sections:

- `memory.embedding_model_name` — sentence-transformers model (default: `all-MiniLM-L6-v2`)
- `experts.default_weights` — base routing weights per layer
- `experts.intent_gating.mode` — `"layered"` or `"expert"`
- `experts.intent_gating.intents` — keyword-based intent detection with boost and target layers
- `deducer.base_profiles` — pre-configured moral profiles per layer type / layer id
- `learning.salience_keywords` / `learning.vector_delta_step` — controls continual learning sensitivity
- `wrapper.local_testing.use_gemma` / `use_lm_studio` / `lm_studio_url` — LLM backend selection
- `wrapper.safety_floor` — hard-coded constraints always prepended to the steering prefix

## LLM backend

`MoralityWrapper.call_llm` accepts any `llm_func(prompt: str) -> str`. Three patterns are shown in `core.py` and `interactive_run.py`:
1. **Mock** — returns `f"Mock Response to: {prompt[:100]}..."` (default when nothing configured)
2. **LM Studio** — OpenAI-compatible HTTP call to local server (`use_lm_studio: true`, configure `lm_studio_url`)
3. **Gemma** — HuggingFace `transformers` pipeline (`use_gemma: true`, requires `HF_TOKEN`)

## Docker

```bash
docker-compose up runtime   # runs opendike-run
docker-compose up training  # runs opendike-train
```
