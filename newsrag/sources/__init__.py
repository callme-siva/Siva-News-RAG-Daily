"""Source adapters. Each one turns a configured source into normalised `Item`s."""

from __future__ import annotations

from newsrag.config import SourceConfig
from newsrag.secrets import KeyStore
from newsrag.sources.base import SourceAdapter
from newsrag.sources.gdelt import GdeltAdapter
from newsrag.sources.keyed import KEYED_ADAPTERS
from newsrag.sources.rss import RSSAdapter


class MissingKey(Exception):
    """A key-based source was configured but its key is not in the KeyStore."""


def build_adapter(config: SourceConfig, keys: KeyStore) -> SourceAdapter:
    if config.type == "rss":
        return RSSAdapter(config)
    if config.type == "gdelt":
        return GdeltAdapter(config)
    assert config.api is not None
    adapter_cls = KEYED_ADAPTERS.get(config.api)
    if adapter_cls is None:
        raise ValueError(f"Unknown API {config.api!r} for source {config.name!r}")
    key = keys.get(config.api)
    if key is None:
        raise MissingKey(config.api)
    return adapter_cls(config, key)
