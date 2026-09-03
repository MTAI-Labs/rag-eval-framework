"""Golden set: conversion from Excel, schema validation and checksummed loading."""

from rag_eval.dataset.refs import ReferenceParseError, parse_reference
from rag_eval.dataset.loader import (
    DatasetError,
    load_golden_set,
    write_golden_set,
    dataset_checksum,
    read_manifest,
)

__all__ = [
    "ReferenceParseError",
    "parse_reference",
    "DatasetError",
    "load_golden_set",
    "write_golden_set",
    "dataset_checksum",
    "read_manifest",
]
