import asyncio
import logging

import fastembed
from datapizza.core.embedder import BaseEmbedder
from datapizza.type import SparseEmbedding

log = logging.getLogger(__name__)


class FastEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        embedding_name: str | None = None,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        if embedding_name:
            self.embedding_name = embedding_name
        else:
            self.embedding_name = model_name

        self.cache_dir = cache_dir
        self.embedder = fastembed.SparseTextEmbedding(
            model_name=model_name, cache_dir=cache_dir
        )

    def embed(self, text: str | list[str], model_name: str | None = None):
        # fastembed.embed() returns an iterable; convert to list to materialize all embeddings
        embeddings = list(self.embedder.embed(text))

        if isinstance(text, list):
            return [
                SparseEmbedding(
                    name=self.embedding_name,
                    values=embedding.values.tolist(),
                    indices=embedding.indices.tolist(),
                )
                for embedding in embeddings
            ]
        else:
            # Single text input returns single embedding
            embedding = embeddings[0]
            return SparseEmbedding(
                name=self.embedding_name,
                values=embedding.values.tolist(),
                indices=embedding.indices.tolist(),
            )

    async def a_embed(self, text: str | list[str], model_name: str | None = None):
        # run the sync embed() in a thread to avoid blocking the event loop
        return await asyncio.to_thread(self.embed, text, model_name)
