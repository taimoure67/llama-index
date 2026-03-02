# Code Explanation: Triple Extractor

A deep-dive into every file in this repository, how each piece works, and how
they fit together to extract knowledge-graph triples from raw text.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Dependencies](#3-dependencies)
4. [Architecture & Data Flow](#4-architecture--data-flow)
5. [Module-by-Module Explanation](#5-module-by-module-explanation)
   - [llm\_config.py](#51-triple_extractorllm_configpy)
   - [embedding\_config.py](#52-triple_extractorembedding_configpy)
   - [prompts.py](#53-triple_extractorpromptspy)
   - [extractor.py](#54-triple_extractorextractorpy)
   - [\_\_init\_\_.py](#55-triple_extractor__init__py)
6. [End-to-End Example Walkthrough](#6-end-to-end-example-walkthrough)
7. [Configuration Reference](#7-configuration-reference)
8. [Customisation Guide](#8-customisation-guide)

---

## 1. Overview

### What is a Triple Extractor?

A **triple extractor** reads unstructured text and produces a list of
*knowledge-graph triples* — structured facts of the form:

```
(Subject, Predicate, Object)
```

For example, from the sentence *"Albert Einstein was born in Ulm, Germany"* a
triple extractor might yield:

```
("Albert Einstein", "was born in", "Ulm")
("Ulm", "is located in", "Germany")
```

These triples are the building blocks of knowledge graphs, enabling downstream
tasks such as question answering, entity linking, and semantic search.

### Technology Choices

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration framework | **LlamaIndex** | Production-grade RAG/graph tooling with built-in chunking, document management, and graph indices |
| LLM | **GPT-OSS-120B** | A 120-billion-parameter open-source-compatible model served via an OpenAI-compatible API; large enough for high-quality relation extraction |
| Embedding model | **IBM Granite** (`ibm/granite-embedding-278m-multilingual`) | Compact, multilingual embedding model available through IBM WatsonX; used by the PropertyGraphIndex to embed graph nodes |

---

## 2. Project Structure

```
llama-index/
├── requirements.txt                   # Python package dependencies
├── CODE_EXPLANATION.md                # This document
├── triple_extractor/                  # Main package
│   ├── __init__.py                    # Public API exports
│   ├── llm_config.py                  # GPT-OSS-120B factory function
│   ├── embedding_config.py            # IBM Granite embedding factory function
│   ├── prompts.py                     # Custom triple-extraction prompt template
│   └── extractor.py                   # TripleExtractor class (core logic)
└── examples/
    └── example.py                     # End-to-end usage script
```

---

## 3. Dependencies

**File:** `requirements.txt`

```
llama-index-core>=0.12.0
llama-index-llms-openai-like>=0.3.0
llama-index-embeddings-ibm>=0.3.0
```

| Package | Role |
|---------|------|
| `llama-index-core` | Core LlamaIndex primitives: `Document`, `PropertyGraphIndex`, `SimpleLLMPathExtractor`, `PromptTemplate`, `Settings`, and the `SimplePropertyGraphStore` in-memory graph store |
| `llama-index-llms-openai-like` | Provides the `OpenAILike` class, which wraps any HTTP endpoint that speaks the OpenAI Chat Completions protocol (used to call GPT-OSS-120B) |
| `llama-index-embeddings-ibm` | Provides `WatsonxEmbeddings`, IBM's official LlamaIndex embedding integration for WatsonX-hosted models including the Granite family |

Install with:

```bash
pip install -r requirements.txt
```

---

## 4. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        User Code                            │
│  extractor.extract(["text1", "text2", ...])                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              Convert strings → Document objects
                         │
                         ▼
         Configure LlamaIndex global Settings
         (LLM, embed_model, chunk_size, chunk_overlap)
                         │
                         ▼
    ┌────────────────────────────────────────┐
    │         PropertyGraphIndex             │
    │  ┌──────────────────────────────────┐  │
    │  │    SimpleLLMPathExtractor        │  │
    │  │  - splits text into chunks       │  │
    │  │  - calls LLM (gpt-oss-120b)     │  │
    │  │    with TRIPLE_EXTRACT_PROMPT    │  │
    │  │  - receives (S, P, O) lines      │  │
    │  └──────────────────────────────────┘  │
    │  ┌──────────────────────────────────┐  │
    │  │    SimplePropertyGraphStore      │  │
    │  │  - stores EntityNode objects     │  │
    │  │  - stores Relation objects       │  │
    │  └──────────────────────────────────┘  │
    └────────────────────┬───────────────────┘
                         │
                         ▼
          _collect_triples(index)
          ├── primary: store.get_triplets() → Relation objects
          └── fallback: _extract_via_prompt() → call LLM + regex
                         │
                         ▼
          _parse_triplets(llm_response)   [fallback only]
          regex: \(\s*([^,)]+?)\s*,\s*([^,)]+?)\s*,\s*([^,)]+?)\s*\)
                         │
                         ▼
         Deduplicate (order-preserving) via dict.fromkeys()
                         │
                         ▼
         list[tuple[str, str, str]]   ← returned to caller
```

---

## 5. Module-by-Module Explanation

### 5.1 `triple_extractor/llm_config.py`

**Purpose:** Instantiate an LLM object that points to the GPT-OSS-120B
endpoint.  All connection details (URL, key) are kept here rather than
scattered across the codebase.

#### `get_llm()` factory

```python
def get_llm(
    model: str = "gpt-oss-120b",
    api_base: str | None = None,
    api_key: str | None = None,
    context_window: int = 128_000,
    is_chat_model: bool = True,
    is_function_calling_model: bool = False,
    max_tokens: int = 1024,
) -> OpenAILike:
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gpt-oss-120b"` | Model name sent in every API request |
| `api_base` | `None` → env `OPENAI_API_BASE` | Base URL of the OpenAI-compatible endpoint (e.g. `https://host/v1`) |
| `api_key` | `None` → env `OPENAI_API_KEY` | Bearer token for authentication |
| `context_window` | `128_000` | Maximum number of tokens the model can see at once |
| `is_chat_model` | `True` | Tells LlamaIndex to use the `/chat/completions` route |
| `is_function_calling_model` | `False` | Disables tool-call formatting that GPT-OSS-120B does not need |
| `max_tokens` | `1024` | Caps the length of each LLM response |

**How it works:**
1. Resolves `api_base` and `api_key` from arguments first; falls back to
   environment variables.
2. Raises `ValueError` if either value is still empty — fail fast before any
   network call is made.
3. Returns an `OpenAILike` object, which LlamaIndex treats identically to a
   native `OpenAI` LLM.

---

### 5.2 `triple_extractor/embedding_config.py`

**Purpose:** Instantiate an IBM Granite embedding model via WatsonX.
Embeddings are used by the `PropertyGraphIndex` to represent graph nodes in
vector space, enabling semantic retrieval later.

#### Model constants

```python
GRANITE_EMBEDDING_107M = "ibm/granite-embedding-107m-multilingual"
GRANITE_EMBEDDING_278M = "ibm/granite-embedding-278m-multilingual"
```

The 278M model (default) produces higher-quality embeddings at the cost of
slightly more compute.  The 107M model is faster and cheaper for
resource-constrained environments.

#### `get_embedding_model()` factory

```python
def get_embedding_model(
    model_id: str = GRANITE_EMBEDDING_278M,
    url: str | None = None,
    project_id: str | None = None,
    apikey: str | None = None,
) -> WatsonxEmbeddings:
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `GRANITE_EMBEDDING_278M` | WatsonX model identifier |
| `url` | `None` → env `WATSONX_URL` | WatsonX regional endpoint (e.g. `https://us-south.ml.cloud.ibm.com`) |
| `project_id` | `None` → env `WATSONX_PROJECT_ID` | WatsonX project that has access to the Granite model |
| `apikey` | `None` → env `WATSONX_API_KEY` | IBM Cloud API key |

Same credential-resolution pattern as `get_llm()`: arguments take priority
over environment variables; missing values raise `ValueError`.

---

### 5.3 `triple_extractor/prompts.py`

**Purpose:** Define the exact instruction given to the LLM when extracting
triples.  A well-crafted prompt is critical — vague instructions produce
unparseable output.

#### Template source

```
You are a knowledge-graph expert. Your task is to extract factual
knowledge triplets from the text provided below.

Rules:
- Each triplet must be on its own line, enclosed in parentheses.
- Format: (subject, predicate, object)
- Use concise, canonical forms (e.g. 'Albert Einstein', not 'he').
- Extract at most {max_knowledge_triplets} triplets.
- Only output triplets – no explanations, no numbering, no extra text.

Text:
-----
{text}
-----

Triplets:
```

#### Template variables

| Variable | Filled by | Description |
|----------|-----------|-------------|
| `{text}` | LlamaIndex / `_extract_via_prompt()` | The text chunk to analyse |
| `{max_knowledge_triplets}` | `TripleExtractor.max_triplets_per_chunk` | Hard upper bound on triples per chunk |

#### Design decisions

- **Parentheses delimiters** make regex parsing reliable — no ambiguity between
  commas used as list separators versus commas inside entity names.
- **"canonical forms" rule** prevents pronouns (`he`, `she`) from becoming
  graph nodes, which would create disconnected islands.
- **"no extra text" rule** minimises the noise the regex must filter out.

---

### 5.4 `triple_extractor/extractor.py`

This is the heart of the package.

#### Module-level regex

```python
_TRIPLET_RE = re.compile(
    r"\(\s*([^,\)]+?)\s*,\s*([^,\)]+?)\s*,\s*([^,\)]+?)\s*\)"
)
```

Breakdown:

| Token | Meaning |
|-------|---------|
| `\(` | Literal opening parenthesis |
| `\s*` | Optional whitespace |
| `([^,\)]+?)` | Capture group — one or more chars that are **not** `,` or `)`, non-greedy |
| `\s*,\s*` | Comma separator with optional surrounding spaces |
| `\)` | Literal closing parenthesis |

This pattern is applied three times (subject, predicate, object) in a single
match, so a line like `(Marie Curie, was born in, Warsaw)` produces three
capture groups: `"Marie Curie"`, `"was born in"`, `"Warsaw"`.

---

#### `TripleExtractor` class

```python
class TripleExtractor:
    def __init__(
        self,
        llm: Any,
        embed_model: Any,
        max_triplets_per_chunk: int = 10,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm` | — | Any LlamaIndex-compatible LLM (e.g. `OpenAILike`) |
| `embed_model` | — | Any LlamaIndex-compatible embedder (e.g. `WatsonxEmbeddings`) |
| `max_triplets_per_chunk` | `10` | Passed to `SimpleLLMPathExtractor` and the prompt |
| `chunk_size` | `512` | Tokens per text chunk; smaller = more LLM calls, finer granularity |
| `chunk_overlap` | `64` | Token overlap between consecutive chunks; preserves context at boundaries |

---

#### `extract(texts)` — public entry point

```python
def extract(self, texts: list[str]) -> list[tuple[str, str, str]]:
```

Wraps each string in a `Document` object and delegates to
`extract_from_documents()`.  This is the most common entry point.

---

#### `extract_from_documents(documents)` — accepts LlamaIndex Documents

```python
def extract_from_documents(
    self, documents: list[Document]
) -> list[tuple[str, str, str]]:
```

Core orchestration method:

1. **Writes to global `Settings`** — LlamaIndex's internal components (node
   parsers, retrievers) read from `Settings` rather than being passed models
   directly.
2. **Creates `SimpleLLMPathExtractor`** — LlamaIndex's built-in extractor that
   calls the LLM on each chunk and parses `(entity, relation, entity)` paths.
3. **Creates `SimplePropertyGraphStore`** — an in-memory graph store.
4. **Calls `PropertyGraphIndex.from_documents()`** — chunks the documents,
   runs the extractor on each chunk, and stores the results in the graph store.
5. **Delegates to `_collect_triples()`** to harvest the results.

---

#### `_collect_triples(index)` — primary collection path

```python
def _collect_triples(
    self, index: PropertyGraphIndex
) -> list[tuple[str, str, str]]:
```

Checks whether the graph store exposes `get_triplets()`:

- **Primary path** (available): iterates over `Relation` objects stored in
  `SimplePropertyGraphStore`, reading `.source_id`, `.label`, and `.target_id`
  to form `(subject, predicate, object)` tuples.
- **Fallback path** (unavailable): calls `_extract_via_prompt()` to re-run the
  LLM using the custom `TRIPLE_EXTRACT_PROMPT` and parse the output with the
  module-level regex.

Deduplication is done with `dict.fromkeys(triples)` which removes duplicates
while preserving insertion order — important for deterministic output.

---

#### `_extract_via_prompt(index)` — LLM fallback

```python
def _extract_via_prompt(
    self, index: PropertyGraphIndex
) -> list[tuple[str, str, str]]:
```

Iterates over every node stored in the index's `docstore`, formats
`TRIPLE_EXTRACT_PROMPT` with the node's text, calls `self.llm.complete()`, and
feeds the response to `_parse_triplets()`.

---

#### `_parse_triplets(text)` — module-level helper

```python
def _parse_triplets(text: str) -> list[tuple[str, str, str]]:
```

Applies `_TRIPLET_RE` to the raw LLM response string and returns all matches
as `(subject, predicate, object)` tuples with leading/trailing whitespace
stripped.

---

### 5.5 `triple_extractor/__init__.py`

Exposes the public API so callers need only one import:

```python
from triple_extractor import TripleExtractor, get_llm, get_embedding_model
```

Exported symbols:

| Symbol | Source module | Description |
|--------|--------------|-------------|
| `TripleExtractor` | `extractor` | Main class |
| `get_llm` | `llm_config` | GPT-OSS-120B factory |
| `get_embedding_model` | `embedding_config` | Granite embedding factory |
| `GRANITE_EMBEDDING_107M` | `embedding_config` | 107M model ID constant |
| `GRANITE_EMBEDDING_278M` | `embedding_config` | 278M model ID constant |

---

## 6. End-to-End Example Walkthrough

**File:** `examples/example.py`

### Step-by-step

```python
# 1. Resolve credentials and create the LLM
llm = get_llm()
```
Reads `OPENAI_API_BASE` and `OPENAI_API_KEY` from the environment, validates
they are non-empty, and returns an `OpenAILike` instance pointed at the
GPT-OSS-120B endpoint.

```python
# 2. Resolve credentials and create the embedding model
embed_model = get_embedding_model()
```
Reads `WATSONX_URL`, `WATSONX_PROJECT_ID`, and `WATSONX_API_KEY` from the
environment and returns a `WatsonxEmbeddings` instance using the 278M Granite
model.

```python
# 3. Build the extractor
extractor = TripleExtractor(llm=llm, embed_model=embed_model)
```
Stores the LLM and embedding model; sets default chunk parameters (512 tokens,
64 overlap, up to 10 triples per chunk).

```python
# 4. Run extraction
triples = extractor.extract(SAMPLE_TEXTS)
```
Under the hood:
- Both strings in `SAMPLE_TEXTS` are wrapped in `Document` objects.
- LlamaIndex splits them into 512-token chunks (with 64-token overlap).
- For each chunk, `SimpleLLMPathExtractor` calls GPT-OSS-120B with the triple
  extraction prompt.
- The LLM returns lines like `(Albert Einstein, was born in, Ulm)`.
- LlamaIndex parses those into `EntityNode` / `Relation` objects stored in
  `SimplePropertyGraphStore`.
- `_collect_triples()` reads them out and deduplicates.

```python
# 5. Print results
for i, (subj, pred, obj) in enumerate(triples, 1):
    print(f"  {i:>3}. ({subj!r}, {pred!r}, {obj!r})")
```
Expected output (actual content depends on the LLM):
```
    1. ('Albert Einstein', 'was born on', 'March 14, 1879')
    2. ('Albert Einstein', 'was born in', 'Ulm')
    3. ('Albert Einstein', 'developed', 'theory of relativity')
    4. ('Albert Einstein', 'received', 'Nobel Prize in Physics')
    5. ('Marie Curie', 'conducted research on', 'radioactivity')
    ...
```

---

## 7. Configuration Reference

| Environment Variable | Module | Required | Description |
|---------------------|--------|----------|-------------|
| `OPENAI_API_BASE` | `llm_config.py` | Yes | Base URL of the GPT-OSS-120B endpoint, e.g. `https://your-host/v1` |
| `OPENAI_API_KEY` | `llm_config.py` | Yes | API key / bearer token for the endpoint |
| `WATSONX_URL` | `embedding_config.py` | Yes | IBM WatsonX service URL, e.g. `https://us-south.ml.cloud.ibm.com` |
| `WATSONX_PROJECT_ID` | `embedding_config.py` | Yes | WatsonX project that has access to the Granite embedding model |
| `WATSONX_API_KEY` | `embedding_config.py` | Yes | IBM Cloud API key |

All values can also be passed directly as arguments to `get_llm()` and
`get_embedding_model()` — this overrides the environment variables.

---

## 8. Customisation Guide

### Swap the LLM model name

```python
llm = get_llm(model="gpt-oss-70b", api_base="...", api_key="...")
```

### Use the smaller (faster) Granite model

```python
from triple_extractor import get_embedding_model, GRANITE_EMBEDDING_107M

embed = get_embedding_model(model_id=GRANITE_EMBEDDING_107M)
```

### Tune chunking and triple density

```python
extractor = TripleExtractor(
    llm=llm,
    embed_model=embed,
    max_triplets_per_chunk=5,   # fewer, higher-precision triples
    chunk_size=256,             # smaller chunks for dense text
    chunk_overlap=32,
)
```

### Plug in a custom prompt

Edit `triple_extractor/prompts.py` and replace `TRIPLE_EXTRACT_TMPL` with your
own template.  The only hard constraint is that the template must accept
`{text}` and `{max_knowledge_triplets}` variables, and the LLM must be
instructed to return lines of the form `(subject, predicate, object)` so that
`_TRIPLET_RE` can parse them.

### Pass pre-built Documents

```python
from llama_index.core import Document

docs = [
    Document(text="...", metadata={"source": "paper-1"}),
    Document(text="...", metadata={"source": "paper-2"}),
]
triples = extractor.extract_from_documents(docs)
```

Use `extract_from_documents()` when you need to attach metadata or when your
documents are already split / loaded by another LlamaIndex reader.
