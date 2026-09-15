"""Storage package providing abstract and concrete persistence implementations."""

from regrag.storage.base import BenchmarkResultRepository
from regrag.storage.in_memory import InMemoryResultRepository
from regrag.storage.file_repo import FileResultRepository

__all__ = [
    "BenchmarkResultRepository",
    "InMemoryResultRepository",
    "FileResultRepository",
]
