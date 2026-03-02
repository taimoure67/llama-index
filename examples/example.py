"""
End-to-end example: extract knowledge-graph triples from sample text.

Required environment variables
-------------------------------
  OPENAI_API_BASE      Base URL for the gpt-oss-120b endpoint
                       e.g. https://your-endpoint.example.com/v1
  OPENAI_API_KEY       API key for the endpoint

  WATSONX_URL          IBM WatsonX service URL
                       e.g. https://us-south.ml.cloud.ibm.com
  WATSONX_PROJECT_ID   WatsonX project ID
  WATSONX_API_KEY      IBM Cloud API key

Usage
-----
  export OPENAI_API_BASE="https://..."
  export OPENAI_API_KEY="sk-..."
  export WATSONX_URL="https://us-south.ml.cloud.ibm.com"
  export WATSONX_PROJECT_ID="<your-project-id>"
  export WATSONX_API_KEY="<your-ibm-cloud-api-key>"

  python examples/example.py
"""

import sys
import os

# Allow running the script from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triple_extractor import TripleExtractor, get_embedding_model, get_llm

# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------
SAMPLE_TEXTS = [
    (
        "Albert Einstein was born on March 14, 1879, in Ulm, in the Kingdom of "
        "Württemberg in the German Empire. He developed the theory of relativity, "
        "one of the two pillars of modern physics. Einstein received the Nobel Prize "
        "in Physics in 1921 for his discovery of the law of the photoelectric effect."
    ),
    (
        "Marie Curie was a Polish and naturalized-French physicist and chemist who "
        "conducted pioneering research on radioactivity. She was the first woman to "
        "win a Nobel Prize, the first person to win the Nobel Prize twice, and the "
        "only person to win the Nobel Prize in two different sciences. She was born "
        "in Warsaw, which was then part of the Russian Empire."
    ),
]


def main() -> None:
    print("Initialising LLM (gpt-oss-120b) …")
    llm = get_llm()  # reads OPENAI_API_BASE / OPENAI_API_KEY from environment

    print("Initialising embedding model (IBM Granite) …")
    embed_model = get_embedding_model()  # reads WATSONX_* from environment

    extractor = TripleExtractor(
        llm=llm,
        embed_model=embed_model,
        max_triplets_per_chunk=10,
    )

    print(f"\nExtracting triples from {len(SAMPLE_TEXTS)} text(s) …\n")
    triples = extractor.extract(SAMPLE_TEXTS)

    if not triples:
        print("No triples were extracted.")
    else:
        print(f"Extracted {len(triples)} triple(s):\n")
        for i, (subj, pred, obj) in enumerate(triples, 1):
            print(f"  {i:>3}. ({subj!r}, {pred!r}, {obj!r})")


if __name__ == "__main__":
    main()
