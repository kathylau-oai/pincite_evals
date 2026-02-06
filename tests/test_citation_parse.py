import sys
from pathlib import Path

# Allow running tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pincite_evals.citations import extract_excerpt_citations


def test_extract_excerpt_citations_basic():
    out = "Use controlling authority (TWOMBLY_2007[P012.B07#A1F3])."
    citations = extract_excerpt_citations(out)
    assert len(citations) == 1
    assert citations[0].doc_id == "TWOMBLY_2007"
    assert citations[0].excerpt_id == "P012.B07#A1F3"
    assert citations[0].raw == "TWOMBLY_2007[P012.B07#A1F3]"
