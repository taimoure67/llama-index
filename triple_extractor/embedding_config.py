"""
Embedding model configuration using IBM Granite via WatsonX.
"""

import os

from llama_index.embeddings.ibm import WatsonxEmbeddings


# Available IBM Granite embedding model IDs
GRANITE_EMBEDDING_107M = "ibm/granite-embedding-107m-multilingual"
GRANITE_EMBEDDING_278M = "ibm/granite-embedding-278m-multilingual"


def get_embedding_model(
    model_id: str = GRANITE_EMBEDDING_278M,
    url: str | None = None,
    project_id: str | None = None,
    apikey: str | None = None,
) -> WatsonxEmbeddings:
    """
    Create a WatsonxEmbeddings instance configured for an IBM Granite model.

    Credentials are read from environment variables if not provided:
      - WATSONX_URL:        IBM WatsonX service URL
      - WATSONX_PROJECT_ID: WatsonX project ID
      - WATSONX_API_KEY:    IBM Cloud API key

    Args:
        model_id: Granite embedding model ID to use.
                  Defaults to "ibm/granite-embedding-278m-multilingual".
        url: WatsonX service URL.
        project_id: WatsonX project ID.
        apikey: IBM Cloud API key.

    Returns:
        Configured WatsonxEmbeddings instance.
    """
    resolved_url = url or os.environ.get("WATSONX_URL", "")
    resolved_project_id = project_id or os.environ.get("WATSONX_PROJECT_ID", "")
    resolved_apikey = apikey or os.environ.get("WATSONX_API_KEY", "")

    if not resolved_url:
        raise ValueError(
            "url must be provided or WATSONX_URL environment variable must be set."
        )
    if not resolved_project_id:
        raise ValueError(
            "project_id must be provided or WATSONX_PROJECT_ID environment variable must be set."
        )
    if not resolved_apikey:
        raise ValueError(
            "apikey must be provided or WATSONX_API_KEY environment variable must be set."
        )

    return WatsonxEmbeddings(
        model_id=model_id,
        url=resolved_url,
        project_id=resolved_project_id,
        apikey=resolved_apikey,
    )
