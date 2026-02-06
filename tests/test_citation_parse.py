import sys
from pathlib import Path

# Allow running tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pincite_evals.citations import extract_excerpt_citations


def test_extract_excerpt_citations_basic():
    out = "Use controlling authority (TWOMBLY_2007[P012.B07])."
    citations = extract_excerpt_citations(out)
    assert len(citations) == 1
    assert citations[0].doc_id == "TWOMBLY_2007"
    assert citations[0].excerpt_id == "P012.B07"
    assert citations[0].raw == "TWOMBLY_2007[P012.B07]"


def test_extract_excerpt_citations_allows_no_hash_format():
    out = "Use controlling authority (TWOMBLY_2007[P012.B07])."
    citations = extract_excerpt_citations(out)
    assert len(citations) == 1
    assert citations[0].raw == "TWOMBLY_2007[P012.B07]"


def test_extract_excerpt_citations_rejects_malformed_excerpt_id():
    out = "Bad cite (TWOMBLY_2007[random_text])."
    citations = extract_excerpt_citations(out)
    assert citations == []


def test_extract_excerpt_citations_rejects_legacy_paragraph_format():
    out = "Legacy cite [TWOMBLY_2007 ¶12-14]."
    citations = extract_excerpt_citations(out)
    assert citations == []


def test_extract_excerpt_citations_rejects_footnote_ids():
    out = "Footnote cite (TWOMBLY_2007[P012.FN02])."
    citations = extract_excerpt_citations(out)
    assert citations == []


def test_extract_excerpt_citations_rejects_hashed_ids():
    out = "Old hash format (TWOMBLY_2007[P012.B07#A1F3])."
    citations = extract_excerpt_citations(out)
    assert citations == []
