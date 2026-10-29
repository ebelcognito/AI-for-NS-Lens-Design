"""Optical context adapters for the official evosax Evolution Transformer.

Metadata utilities remain dependency-free. Model/emitter imports are deliberately lazy
so the default CMA installation does not require JAX or Flax.
"""
from .metadata import (GENE_REGISTRY, encode_environment, encode_observations,
                       parameter_metadata)

__all__ = [
    "GENE_REGISTRY", "encode_environment", "encode_observations",
    "parameter_metadata",
]
