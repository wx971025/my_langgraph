import os
from enum import StrEnum
from langchain_openai import AzureOpenAIEmbeddings as BaseAzureOpenAIEmbeddings


class AzureOpenAIEmbeddingModelName(StrEnum):
    text_embedding_3_small = "text-embedding-3-small"
    text_embedding_3_large = "text-embedding-3-large"

class AzureOpenAIEmbeddings(BaseAzureOpenAIEmbeddings):
    def __init__(self, model: AzureOpenAIEmbeddingModelName):
        super().__init__(
            model=model,
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        )
