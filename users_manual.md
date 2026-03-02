# Users Manual: Triple Extractor

## Table of Contents

1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Quick Start](#5-quick-start)
6. [Usage Guide](#6-usage-guide)
7. [Understanding the Output](#7-understanding-the-output)
8. [Configuration Options](#8-configuration-options)
9. [Running the Bundled Example](#9-running-the-bundled-example)
10. [Troubleshooting](#10-troubleshooting)
11. [FAQ](#11-faq)

---

## 1. Introduction

**Triple Extractor** reads unstructured text and turns it into structured,
machine-readable facts called *knowledge-graph triples*. Each triple is a
three-part statement:

```
(Subject,  Predicate,  Object)
("Marie Curie",  "was born in",  "Warsaw")
```

Feed in a paragraph, a whole document, or a batch of files. Get back a clean
Python list of tuples you can load into a graph database, run graph analytics
on, or feed into a downstream question-answering pipeline.

**Who is this for?**

- Data engineers building knowledge-graph pipelines
- Researchers who need to structure large text corpora
- Developers adding semantic-search or entity-linking features to an application

**What powers it?**

| Role | Technology |
|------|-----------|
| Large language model | GPT-OSS-120B (OpenAI-compatible endpoint) |
| Embedding model | IBM Granite via WatsonX |
| Orchestration | LlamaIndex `PropertyGraphIndex` |

---

## 2. Prerequisites

Before you begin, make sure you have:

- **Python 3.10 or higher**
- **Git** (to clone the repository)
- Access to a **GPT-OSS-120B** endpoint that speaks the OpenAI Chat
  Completions API (your API base URL and key)
- An **IBM Cloud** account with a **WatsonX** project that has the IBM Granite
  embedding models enabled (your WatsonX URL, project ID, and IBM Cloud API key)

---

## 3. Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd llama-index

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The package is not yet published to PyPI. You must run it directly
> from the cloned repository.

---

## 4. Configuration

The extractor reads credentials from **environment variables**. Set all five
before running any code.

| Variable | Where to get it | Description |
|----------|----------------|-------------|
| `OPENAI_API_BASE` | Your GPT-OSS-120B provider | Base URL of the endpoint, e.g. `https://your-host/v1` |
| `OPENAI_API_KEY` | Your GPT-OSS-120B provider | Bearer token / API key |
| `WATSONX_URL` | IBM Cloud → WatsonX console | Regional service URL, e.g. `https://us-south.ml.cloud.ibm.com` |
| `WATSONX_PROJECT_ID` | IBM Cloud → WatsonX project settings | The UUID of your WatsonX project |
| `WATSONX_API_KEY` | IBM Cloud → Manage → API keys | Your IBM Cloud API key |

**Set them in your shell:**

```bash
export OPENAI_API_BASE="https://your-gpt-endpoint/v1"
export OPENAI_API_KEY="sk-your-key-here"

export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
export WATSONX_PROJECT_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
export WATSONX_API_KEY="your-ibm-cloud-api-key"
```

You can also add these lines to your `.bashrc` / `.zshrc` so they persist
across terminal sessions, or store them in a `.env` file and load with
`python-dotenv`.

---

## 5. Quick Start

```python
from triple_extractor import TripleExtractor, get_llm, get_embedding_model

llm   = get_llm()             # reads OPENAI_API_BASE + OPENAI_API_KEY
embed = get_embedding_model() # reads WATSONX_URL + WATSONX_PROJECT_ID + WATSONX_API_KEY

extractor = TripleExtractor(llm=llm, embed_model=embed)
triples   = extractor.extract(["Marie Curie was born in Warsaw, Poland."])

print(triples)
# [('Marie Curie', 'was born in', 'Warsaw'), ('Warsaw', 'is in', 'Poland')]
```

That is all you need to get triples from a text string.

---

## 6. Usage Guide

### 6a. Extract from raw strings

Pass a list of plain-text strings to `extract()`. Each string is treated as a
separate document.

```python
from triple_extractor import TripleExtractor, get_llm, get_embedding_model

llm   = get_llm()
embed = get_embedding_model()
extractor = TripleExtractor(llm=llm, embed_model=embed)

texts = [
    "The Eiffel Tower is located in Paris, France.",
    "Paris is the capital of France and sits on the Seine river.",
]

triples = extractor.extract(texts)
for subj, pred, obj in triples:
    print(f"  {subj!r} --[{pred}]--> {obj!r}")
```

---

### 6b. Extract from files

Use LlamaIndex's `SimpleDirectoryReader` to load files from disk, then pass
the resulting `Document` objects to `extract_from_documents()`.

```python
from llama_index.core import SimpleDirectoryReader
from triple_extractor import TripleExtractor, get_llm, get_embedding_model

llm   = get_llm()
embed = get_embedding_model()
extractor = TripleExtractor(llm=llm, embed_model=embed)

# Load all .txt and .pdf files from a directory
documents = SimpleDirectoryReader("./my_documents").load_data()

triples = extractor.extract_from_documents(documents)
print(f"Extracted {len(triples)} triples from {len(documents)} document(s).")
```

`SimpleDirectoryReader` supports `.txt`, `.pdf`, `.docx`, `.csv`, and more
out of the box.

---

### 6c. Extract from a single file

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_files=["report.pdf"]).load_data()
triples   = extractor.extract_from_documents(documents)
```

---

### 6d. Add metadata to documents

When you need to track which source a triple came from, attach metadata to
your `Document` objects before extraction.

```python
from llama_index.core import Document
from triple_extractor import TripleExtractor, get_llm, get_embedding_model

llm   = get_llm()
embed = get_embedding_model()
extractor = TripleExtractor(llm=llm, embed_model=embed)

docs = [
    Document(text="The Amazon river flows through Brazil.",
             metadata={"source": "geography-101.txt", "page": 1}),
    Document(text="Brazil is the largest country in South America.",
             metadata={"source": "geography-101.txt", "page": 2}),
]

triples = extractor.extract_from_documents(docs)
```

---

## 7. Understanding the Output

`extract()` and `extract_from_documents()` both return:

```python
list[tuple[str, str, str]]
```

Each tuple is `(subject, predicate, object)`:

```python
[
    ("Albert Einstein", "was born in",  "Ulm"),
    ("Albert Einstein", "developed",    "theory of relativity"),
    ("Albert Einstein", "received",     "Nobel Prize in Physics"),
    ("Nobel Prize in Physics", "awarded in", "1921"),
]
```

**Key properties of the output:**

- **Ordered** — triples appear in the order they were found in the text.
- **Deduplicated** — if the same `(S, P, O)` triple appears in multiple chunks
  it is included only once.
- **Paraphrase variants are kept** — `("Einstein", "born in", "Ulm")` and
  `("Albert Einstein", "was born in", "Ulm")` are considered different triples
  because they are not string-identical. The LLM usually resolves pronouns and
  uses canonical names thanks to the prompt, so duplicates of this kind are
  rare.

---

## 8. Configuration Options

All options are set when you create the `TripleExtractor` instance:

```python
extractor = TripleExtractor(
    llm=llm,
    embed_model=embed,
    max_triplets_per_chunk=10,   # default
    chunk_size=512,              # default
    chunk_overlap=64,            # default
)
```

| Option | Default | Effect |
|--------|---------|--------|
| `max_triplets_per_chunk` | `10` | Maximum triples the LLM extracts per chunk. **Lower** (e.g. `5`) = higher precision, fewer noisy facts. **Higher** (e.g. `20`) = better recall on dense text. |
| `chunk_size` | `512` | Tokens per text chunk. **Smaller** chunks suit highly technical or information-dense text. **Larger** chunks reduce the number of LLM calls but may miss facts split across sentences. |
| `chunk_overlap` | `64` | Tokens shared between consecutive chunks. **Increase** if you notice that triples spanning a chunk boundary are being missed. |

### Switching the Granite embedding model

The default model is `ibm/granite-embedding-278m-multilingual` (278 M
parameters, higher quality). The 107 M variant is faster and uses less memory:

```python
from triple_extractor import get_embedding_model, GRANITE_EMBEDDING_107M

embed = get_embedding_model(model_id=GRANITE_EMBEDDING_107M)
```

---

## 9. Running the Bundled Example

A ready-to-run script is included at `examples/example.py`. It extracts
triples from two short biographies (Einstein and Curie).

```bash
# From the repo root, with all five env vars set:
python examples/example.py
```

Expected terminal output:

```
Initialising LLM (gpt-oss-120b) …
Initialising embedding model (IBM Granite) …

Extracting triples from 2 text(s) …

Extracted 12 triple(s):

    1. ('Albert Einstein', 'was born on', 'March 14, 1879')
    2. ('Albert Einstein', 'was born in', 'Ulm')
    3. ('Ulm', 'is in', 'Kingdom of Württemberg')
    4. ('Albert Einstein', 'developed', 'theory of relativity')
    5. ('Albert Einstein', 'received', 'Nobel Prize in Physics')
    6. ('Nobel Prize in Physics', 'received in', '1921')
    7. ('Marie Curie', 'was', 'physicist and chemist')
    8. ('Marie Curie', 'conducted research on', 'radioactivity')
    9. ('Marie Curie', 'was first woman to win', 'Nobel Prize')
   10. ('Marie Curie', 'won Nobel Prize', 'twice')
   11. ('Marie Curie', 'was born in', 'Warsaw')
   12. ('Warsaw', 'was part of', 'Russian Empire')
```

> The exact triples depend on the LLM response; your output may differ slightly.

---

## 10. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ValueError: api_base must be provided or OPENAI_API_BASE environment variable must be set.` | `OPENAI_API_BASE` is not exported | Run `export OPENAI_API_BASE="https://..."` |
| `ValueError: api_key must be provided or OPENAI_API_KEY environment variable must be set.` | `OPENAI_API_KEY` is not exported | Run `export OPENAI_API_KEY="sk-..."` |
| `ValueError: url must be provided or WATSONX_URL environment variable must be set.` | `WATSONX_URL` is not exported | Run `export WATSONX_URL="https://..."` |
| `ValueError: project_id must be provided …` | `WATSONX_PROJECT_ID` is not exported | Run `export WATSONX_PROJECT_ID="uuid"` |
| `ValueError: apikey must be provided …` | `WATSONX_API_KEY` is not exported | Run `export WATSONX_API_KEY="..."` |
| `ModuleNotFoundError: No module named 'llama_index'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'llama_index.llms.openai_like'` | Incomplete install | Run `pip install llama-index-llms-openai-like` |
| Result is an empty list `[]` | LLM output was not parseable or the endpoint returned an error | Check the endpoint URL and key; increase `max_tokens` in `get_llm()` |
| Triples look like `('he', 'was', 'scientist')` | LLM ignored the canonical-forms rule | Verify the endpoint is running GPT-OSS-120B; a weaker model may ignore prompt constraints |
| Very slow extraction | Large documents being chunked into many pieces | Increase `chunk_size` or pre-split documents yourself |
| `ConnectionError` / `TimeoutError` | Network issue reaching the endpoint | Check firewall rules; confirm the endpoint is reachable with `curl` |

---

## 11. FAQ

**Can I use a different LLM?**

Yes. Any LlamaIndex-compatible LLM works. For example, to use a local Ollama
model:

```python
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3", request_timeout=120.0)
extractor = TripleExtractor(llm=llm, embed_model=embed)
```

**Can I use a different embedding model?**

Yes. Pass any LlamaIndex-compatible embedder as `embed_model`. For example,
to use a local HuggingFace model:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
extractor = TripleExtractor(llm=llm, embed_model=embed)
```

**Is there a rate limit?**

Rate limits are imposed by your endpoint provider, not by this package. If you
hit rate limits, reduce parallelism by processing smaller batches of documents.

**How do I save the triples to a file?**

```python
import json

triples = extractor.extract(texts)

# Save as JSON array of [subject, predicate, object] arrays
with open("triples.json", "w") as f:
    json.dump(triples, f, indent=2)

# Or as CSV
import csv
with open("triples.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "predicate", "object"])
    writer.writerows(triples)
```

**Does it work on non-English text?**

Yes. The default embedding model (`ibm/granite-embedding-278m-multilingual`)
supports multiple languages. The quality of the extracted triples depends on
GPT-OSS-120B's multilingual capabilities; results may vary by language.

**How do I load the triples into a graph database?**

Most graph databases accept triples or edges as `(source, relation, target)`
tuples. Example with NetworkX:

```python
import networkx as nx

G = nx.DiGraph()
for subj, pred, obj in triples:
    G.add_edge(subj, obj, label=pred)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
```

**Where can I find more technical details about the implementation?**

See `CODE_EXPLANATION.md` in the repository root for a full developer-oriented
deep-dive into every module, class, and method.
