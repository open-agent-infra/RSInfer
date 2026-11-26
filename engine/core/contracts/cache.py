"""Published model-state references crossing frontend, scheduler and worker boundaries."""

import msgspec


class CacheRef(msgspec.Struct, frozen=True):
    key: str
    tokens: int


class CacheSave(msgspec.Struct, frozen=True):
    row: int
    ref: CacheRef
