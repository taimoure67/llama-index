"""
TripleExtractor: extracts (subject, predicate, object) triples from text
using GPT-OSS-120B as the LLM and IBM Granite as the embedding model.
"""

from __future__ import annotations

import re
from typing import Any

from llama_index.core import Document, Settings
from llama_index.core.indices.property_graph import (
    PropertyGraphIndex,
    SimpleLLMPathExtractor,
)
from llama_index.core.graph_stores import SimplePropertyGraphStore

from .prompts import TRIPLE_EXTRACT_PROMPT


# Regex that matches a single triplet line: (subject, predicate, object)
_TRIPLET_RE = re.compile(
    r"\(\s*([^,\)]+?)\s*,\s*([^,\)]+?)\s*,\s*([^,\)]+?)\s*\)"
)


class TripleExtractor:
    """
    Extracts knowledge-graph triples from plain text using a LlamaIndex
    PropertyGraphIndex backed by SimpleLLMPathExtractor.

    Parameters
    ----------
    llm:
        A LlamaIndex-compatible LLM instance (e.g. OpenAILike configured
        for gpt-oss-120b).
    embed_model:
        A LlamaIndex-compatible embedding model instance (e.g.
        WatsonxEmbeddings configured for an IBM Granite model).
    max_triplets_per_chunk:
        Maximum number of triplets the LLM should extract per text chunk.
    chunk_size:
        Token budget for each text chunk sent to the LLM.
    chunk_overlap:
        Overlap in tokens between consecutive chunks.
    """

    def __init__(
        self,
        llm: Any,
        embed_model: Any,
        max_triplets_per_chunk: int = 10,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.llm = llm
        self.embed_model = embed_model
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, texts: list[str]) -> list[tuple[str, str, str]]:
        """
        Extract triples from a list of raw text strings.

        Args:
            texts: List of plain-text strings to process.

        Returns:
            Deduplicated list of (subject, predicate, object) tuples.
        """
        documents = [Document(text=t) for t in texts]
        return self.extract_from_documents(documents)

    def extract_from_documents(
        self, documents: list[Document]
    ) -> list[tuple[str, str, str]]:
        """
        Extract triples from pre-built LlamaIndex Document objects.

        Args:
            documents: List of LlamaIndex Document objects.

        Returns:
            Deduplicated list of (subject, predicate, object) tuples.
        """
        # Configure global LlamaIndex settings so internal components
        # (node parsers, retrievers, etc.) pick up the right models.
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

        kg_extractor = SimpleLLMPathExtractor(
            llm=self.llm,
            max_paths_per_chunk=self.max_triplets_per_chunk,
            num_workers=1,
        )

        graph_store = SimplePropertyGraphStore()

        index = PropertyGraphIndex.from_documents(
            documents,
            llm=self.llm,
            embed_model=self.embed_model,
            kg_extractors=[kg_extractor],
            property_graph_store=graph_store,
            show_progress=False,
        )

        return self._collect_triples(index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_triples(
        self, index: PropertyGraphIndex
    ) -> list[tuple[str, str, str]]:
        """
        Walk the property graph store and collect all (subject, predicate,
        object) triples that were extracted.
        """
        store = index.property_graph_store
        triples: list[tuple[str, str, str]] = []

        # SimplePropertyGraphStore exposes all relations via get_triplets()
        # which returns Relation objects.  Fall back to parsing the raw
        # LLM response when that is not available.
        if hasattr(store, "get_triplets"):
            for rel in store.get_triplets():
                subj = str(rel.source_id)
                pred = str(rel.label)
                obj = str(rel.target_id)
                triples.append((subj, pred, obj))
        else:
            # Fallback: re-run the LLM on each node's text and parse manually
            triples = self._extract_via_prompt(index)

        return list(dict.fromkeys(triples))  # deduplicate while preserving order

    def _extract_via_prompt(
        self, index: PropertyGraphIndex
    ) -> list[tuple[str, str, str]]:
        """
        Fallback: call the LLM directly with the custom extraction prompt and
        parse (subject, predicate, object) lines from the response.
        """
        triples: list[tuple[str, str, str]] = []
        nodes = index.docstore.docs.values() if index.docstore else []

        for node in nodes:
            text = getattr(node, "text", None) or getattr(node, "get_content", lambda: "")()
            if not text:
                continue

            prompt_str = TRIPLE_EXTRACT_PROMPT.format(
                text=text,
                max_knowledge_triplets=self.max_triplets_per_chunk,
            )
            response = self.llm.complete(prompt_str)
            triples.extend(_parse_triplets(str(response)))

        return triples


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _parse_triplets(text: str) -> list[tuple[str, str, str]]:
    """
    Parse all (subject, predicate, object) triplets from a raw LLM response.

    Handles both explicit parentheses format and bare comma-separated lines.
    """
    triples: list[tuple[str, str, str]] = []

    for match in _TRIPLET_RE.finditer(text):
        subj, pred, obj = match.group(1), match.group(2), match.group(3)
        triples.append((subj.strip(), pred.strip(), obj.strip()))

    return triples
