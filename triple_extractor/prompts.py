"""
Prompt templates for knowledge-graph triple extraction.
"""

from llama_index.core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Custom triple-extraction prompt
# ---------------------------------------------------------------------------
# The LLM is asked to return a numbered list of triples enclosed in
# parentheses: (subject, predicate, object).  The extractor parses this
# format to reconstruct structured triples.

TRIPLE_EXTRACT_TMPL = (
    "You are a knowledge-graph expert. Your task is to extract factual "
    "knowledge triplets from the text provided below.\n\n"
    "Rules:\n"
    "- Each triplet must be on its own line, enclosed in parentheses.\n"
    "- Format: (subject, predicate, object)\n"
    "- Use concise, canonical forms (e.g. 'Albert Einstein', not 'he').\n"
    "- Extract at most {max_knowledge_triplets} triplets.\n"
    "- Only output triplets – no explanations, no numbering, no extra text.\n\n"
    "Text:\n"
    "-----\n"
    "{text}\n"
    "-----\n\n"
    "Triplets:"
)

TRIPLE_EXTRACT_PROMPT = PromptTemplate(TRIPLE_EXTRACT_TMPL)
