"""
triple_extractor
================
Knowledge-graph triple extractor powered by LlamaIndex.

- LLM:        GPT-OSS-120B via an OpenAI-compatible endpoint
- Embeddings: IBM Granite via IBM WatsonX

Quick start
-----------
>>> from triple_extractor import TripleExtractor, get_llm, get_embedding_model
>>>
>>> llm = get_llm()          # reads OPENAI_API_BASE / OPENAI_API_KEY from env
>>> embed = get_embedding_model()  # reads WATSONX_* from env
>>>
>>> extractor = TripleExtractor(llm=llm, embed_model=embed)
>>> triples = extractor.extract(["Marie Curie was born in Warsaw, Poland."])
>>> print(triples)
[('Marie Curie', 'was born in', 'Warsaw'), ('Warsaw', 'is in', 'Poland')]
"""

from .embedding_config import GRANITE_EMBEDDING_107M, GRANITE_EMBEDDING_278M, get_embedding_model
from .extractor import TripleExtractor
from .llm_config import get_llm

__all__ = [
    "TripleExtractor",
    "get_llm",
    "get_embedding_model",
    "GRANITE_EMBEDDING_107M",
    "GRANITE_EMBEDDING_278M",
]
