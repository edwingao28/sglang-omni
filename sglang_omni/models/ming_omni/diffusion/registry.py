# SPDX-License-Identifier: Apache-2.0
"""Lazy diffusion-backend registry for Ming-Omni image generation.

Backends register a name with a zero-arg loader callable that performs the heavy
import lazily, so importing this module (or ``sglang_omni``) never pulls in
diffusers / sensenova_u1 / other torch-heavy backend modules. ``get`` invokes
the loader on demand, mirroring the previous ``_create_backend`` dispatch.
"""

from __future__ import annotations

from collections.abc import Callable

from sglang_omni.models.ming_omni.diffusion.backend import DiffusionBackend

BackendLoader = Callable[[], DiffusionBackend]

_REGISTRY: dict[str, BackendLoader] = {}


def register(name: str, loader: BackendLoader) -> None:
    """Register ``loader`` (a zero-arg factory) under ``name``."""
    _REGISTRY[name] = loader


def get(name: str) -> DiffusionBackend:
    """Instantiate the backend registered under ``name``.

    Raises ``ValueError`` listing the currently-available backends when the
    name is unknown.
    """
    loader = _REGISTRY.get(name)
    if loader is None:
        raise ValueError(
            f"Unknown dit_type: {name!r}. "
            f"Must be one of: {', '.join(available())}."
        )
    return loader()


def available() -> list[str]:
    """Return the sorted list of registered backend names."""
    return sorted(_REGISTRY)


def _load_zimage() -> DiffusionBackend:
    from sglang_omni.models.ming_omni.diffusion.zimage_backend import ZImageBackend

    return ZImageBackend()


def _load_sensenovau1() -> DiffusionBackend:
    from sglang_omni.models.ming_omni.diffusion.sensenovau1_backend import (
        SenseNovaU1Backend,
    )

    return SenseNovaU1Backend()


register("zimage", _load_zimage)
register("sensenovau1", _load_sensenovau1)
