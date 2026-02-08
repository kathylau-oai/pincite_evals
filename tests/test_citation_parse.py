from pincite_evals.citations import extract_excerpt_citations


def test_extract_excerpt_citations_basic():
    out = "Use controlling authority (DOC001.P012.B07)."
    citations = extract_excerpt_citations(out)
    assert len(citations) == 1
    assert citations[0].doc_id == "DOC001"
    assert citations[0].excerpt_id == "P012.B07"
    assert citations[0].raw == "DOC001.P012.B07"


def test_extract_excerpt_citations_rejects_bracket_format():
    out = "Bracket cite (DOC001[P012.B07])."
    citations = extract_excerpt_citations(out)
    assert citations == []


def test_extract_excerpt_citations_rejects_malformed_excerpt_id():
    out = "Bad cite (DOC001.random_text)."
    citations = extract_excerpt_citations(out)
    assert citations == []


def test_extract_excerpt_citations_rejects_legacy_paragraph_format():
    out = "Legacy cite [DOC001 ¶12-14]."
    citations = extract_excerpt_citations(out)
    assert citations == []


def test_extract_excerpt_citations_rejects_footnote_ids():
    out = "Footnote cite (DOC001.P012.FN02)."
    citations = extract_excerpt_citations(out)
    assert citations == []


def test_extract_excerpt_citations_rejects_hashed_ids():
    out = "Old hash format (DOC001.P012.B07#A1F3)."
    citations = extract_excerpt_citations(out)
    assert citations == []
