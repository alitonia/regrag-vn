"""Core data models for RegRAG-VN."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from regrag.provenance import CORPUS_UNSET


@dataclass
class LegalCitation:
    """Represents a formal citation to a Vietnamese legal provision."""
    doc_id: str             # e.g., "18/2024/TT-NHNN"
    article_id: str         # e.g., "15" (Điều 15)
    clause_id: Optional[str] = None  # e.g., "2" (Khoản 2)
    point_id: Optional[str] = None   # e.g., "a" (Điểm a)

    def to_citation_string(self) -> str:
        parts = []
        if self.point_id:
            parts.append(f"Điểm {self.point_id}")
        if self.clause_id:
            parts.append(f"Khoản {self.clause_id}")
        parts.append(f"Điều {self.article_id}")
        parts.append(self.doc_id)
        return " ".join(parts)


@dataclass
class LegalChunk:
    """A granular chunk of a legal document (segmented at Khoản / Clause level)."""
    chunk_id: str
    doc_id: str             # e.g., "18/2024/TT-NHNN"
    doc_title: str          # Title of the Circular/Decree
    chapter: Optional[str]  # e.g., "Chương II"
    article_id: str         # e.g., "14"
    article_title: str      # e.g., "Hạn mức giao dịch thẻ"
    clause_id: Optional[str] = None
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Which corpus produced this chunk. CORPUS_TIER1 rows are development
    # fixtures and must never reach a paper table; see provenance.py.
    corpus_source: str = CORPUS_UNSET

    def formatted_context(self) -> str:
        """Context string with legal hierarchical header for retrieval."""
        header = f"[{self.doc_id}] Điều {self.article_id}. {self.article_title}"
        if self.clause_id:
            header += f" (Khoản {self.clause_id})"
        return f"{header}\n{self.text}"


@dataclass
class GoldQuestion:
    """A curated question in the benchmark set.

    Derived only from the trusted QA CSV. gold_doc_ids are CANONICAL instrument
    ids (e.g. "06/2019/TT-NHNN"), never raw URLs, and are resolved from
    doc_link via corpus.canonical - only 2/65 gold passages name their own
    instrument, so the prose can never be the source of doc_id.

    Two classes share this dataclass and must not be conflated:

    * ``is_answerable=True`` / ``category="factual"`` - has a doc_link and a
      verbatim gold passage; scored for citation accuracy and correctness.
    * ``is_answerable=False`` / ``category="unanswerable"`` - an abstention
      probe. ``gold_doc_ids``, ``gold_citations``, ``gold_passage`` and
      ``source_urls`` are all EMPTY on purpose (nothing is unresolved; there is
      no instrument), and ``doc_id_confidence`` is ``"n/a-unanswerable"``. Such
      rows are excluded from every gold-passage invariant and from the
      unresolved-doc_id tally, but are always kept and reported.
    """
    id: str                 # e.g., "Q001"
    question: str
    is_answerable: bool
    gold_doc_ids: List[str] = field(default_factory=list)
    gold_citations: List[Dict[str, Any]] = field(default_factory=list)
    reference_answer: str = ""
    category: str = "factual"  # "factual", "synthesis", "definition", "unanswerable"
    # Verbatim `text_contains_answer_in_the_doc` cell: the coverage-test target
    # and the Tier 1 chunk source. 61/65 answerable rows name an Điều (the 4
    # exceptions quote a Phụ lục annex table); line-initial clause numbering
    # survives in only 6/65, so clause-level gold is optional. Empty by design
    # on unanswerable probes.
    gold_passage: str = ""
    source_urls: List[str] = field(default_factory=list)
    # manifest | slug | heuristic | passage-named | ambiguous-multi-url |
    # unresolved | n/a-unanswerable (probes: there is no instrument to resolve)
    doc_id_confidence: str = "unresolved"
    author: str = ""

    @property
    def doc_ids_resolved(self) -> bool:
        return bool(self.gold_doc_ids) and not any(
            d.startswith("UNRESOLVED") for d in self.gold_doc_ids
        )


@dataclass
class RetrievedResult:
    """Retrieved document chunk with score and rank."""
    chunk: LegalChunk
    score: float
    rank: int
    # Which backend actually produced this ranking. "DEGRADED:<reason>" means a
    # dependency was missing and the scores are not real; see provenance.py.
    retriever_backend: str = CORPUS_UNSET


@dataclass
class GenerationResult:
    """Model output for a given question and retrieval configuration."""
    question_id: str
    model_name: str
    retrieval_mode: str     # "closed_book", "rag_bm25", "rag_dense", ...
    prompt: str
    raw_response: str
    answer_text: str
    extracted_citations: List[Dict[str, Any]] = field(default_factory=list)
    abstained: bool = False
    # Provenance of the corpus this generation was conditioned on.
    corpus_source: str = CORPUS_UNSET
    retriever_backend: str = CORPUS_UNSET
    # Cache key for idempotent re-runs when the CSV changes under review.
    cache_key: str = ""


@dataclass
class EvaluationRecord:
    """Evaluation metrics for a single generation output."""
    question_id: str
    model_name: str
    retrieval_mode: str
    is_answerable: bool
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    abstained: bool = False
    abstained_correctly: bool = False
    correctness_score: float = 0.0  # 0.0, 1.0, 2.0
    hallucinated: bool = False
    notes: str = ""
    corpus_source: str = CORPUS_UNSET
    retriever_backend: str = CORPUS_UNSET
    # "placeholder" means the scorer for this field is not yet implemented.
    # Placeholders must be loud: a null plus this tag, never a plausible number.
    metric_status: str = "unset"
    placeholder_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
