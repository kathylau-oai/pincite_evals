import sys
from pathlib import Path

# Allow running tests without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from pincite_evals.citations import extract_citations


def test_extract_citations_basic():
    out = "Rule statement [twombly_2007 ¶12-14]."
    cites = extract_citations(out)
    assert len(cites) == 1
    assert cites[0].doc_id == 'twombly_2007'
    assert cites[0].start == 12
    assert cites[0].end == 14
