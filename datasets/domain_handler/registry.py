from __future__ import annotations
from typing import Dict, Type
from .base import DomainHandler
from .libero_hdf5 import LiberoHDF5Handler
from .libero_z import LiberoZHandler

# Registry for dataset handlers
_REGISTRY: Dict[str, Type[DomainHandler]] = {
    # LIBERO (original HDF5 format)
    "libero_hdf5": LiberoHDF5Handler,
    "libero_10": LiberoHDF5Handler,
    "libero_90": LiberoHDF5Handler,
    "libero_goal": LiberoHDF5Handler,
    "libero_object": LiberoHDF5Handler,
    "libero_spatial": LiberoHDF5Handler,
    # LIBERO with latent-action (z) targets for flow-expert pretraining
    "libero_z": LiberoZHandler,
}


def get_handler_cls(dataset_name: str) -> Type[DomainHandler]:
    """Strict lookup: require explicit registration."""
    try:
        return _REGISTRY[dataset_name]
    except KeyError:
        raise KeyError(
            f"No handler registered for dataset '{dataset_name}'. "
            f"Add it to _REGISTRY in datasets/domain_handler/registry.py."
        )
