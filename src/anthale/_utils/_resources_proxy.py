from __future__ import annotations

from typing import Any
from typing_extensions import override

from ._proxy import LazyProxy


class ResourcesProxy(LazyProxy[Any]):
    """A proxy for the `anthale.resources` module.

    This is used so that we can lazily import `anthale.resources` only when
    needed *and* so that users can just import `anthale` and reference `anthale.resources`
    """

    @override
    def __load__(self) -> Any:
        import importlib

        mod = importlib.import_module("anthale.resources")
        return mod


resources = ResourcesProxy().__as_proxied__()
