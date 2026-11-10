"""
tests/test_chunker_tables.py - Unit tests for SPEC-007 & SPEC-025: Structural Tabular Serialization & Atomicity
"""

import pytest
from alcuin.chunker.chunker import chunk_document_text, serialize_tabular_data


def test_serialize_markdown_two_column_table():
    text = (
        "Sec. 6-132. Penalties.\n\n"
        "| Offense | Fine |\n"
        "| :--- | :--- |\n"
        "| First offense | $20.00 |\n"
        "| Second offense | $50.00 |\n"
        "| Third offense | $75.00 |\n"
    )
    result = serialize_tabular_data(text)
    assert "<!-- TABLE_DATA:" in result
    assert '"First offense": "$20.00"' in result
    assert '"Second offense": "$50.00"' in result
    assert '"Third offense": "$75.00"' in result
    # Original table remains visually intact
    assert "| Third offense | $75.00 |" in result


def test_serialize_markdown_multi_column_table():
    text = (
        "| Step | Department | Days | Bond |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| 1 | Animal Control | 5 | $25 |\n"
        "| 2 | Housing Inspector | 10 | $50 |\n"
    )
    result = serialize_tabular_data(text)
    # SPEC-025 Phase 2: Multi-column tables (> 2 columns) remain pure pipe tables
    # without vector-diluting JSON blobs or repetitive cards
    assert "| 1 | Animal Control | 5 | $25 |" in result
    assert "| 2 | Housing Inspector | 10 | $50 |" in result
    assert "<!-- TABLE_DATA:" not in result
    assert "Key-Value Extraction:" not in result


def test_serialize_schedule_text_table():
    text = (
        "Escalating citation penalties for unrestrained animals:\n\n"
        "Offense Penalty\n"
        "1st     $ 20.00 fine\n"
        "2nd     $ 50.00 fine\n"
        "3rd     $ 75.00 fine\n"
        "4th     $ 100.00 fine\n"
        "5th     $ 250.00 fine and seizure of the animal\n"
    )
    result = serialize_tabular_data(text)
    assert "<!-- TABLE_DATA:" in result
    assert '"1st": "$ 20.00 fine"' in result
    assert '"3rd": "$ 75.00 fine"' in result
    assert '"5th": "$ 250.00 fine and seizure of the animal"' in result


def test_serialize_ignores_non_table_text():
    text = (
        "Sec. 8-117. Definitions.\n"
        "Deteriorated means that repair costs do not exceed fifty percent (50%) of the value.\n"
        "Dilapidated means that repair costs exceed fifty percent (50%) of the value.\n"
    )
    result = serialize_tabular_data(text)
    assert "<!-- TABLE_DATA:" not in result
    assert result == text


def test_serialize_idempotent():
    text = (
        "| Offense | Fine |\n"
        "| :--- | :--- |\n"
        "| First offense | $20.00 |\n"
    )
    res1 = serialize_tabular_data(text)
    res2 = serialize_tabular_data(res1)
    assert res1 == res2
    assert res2.count("<!-- TABLE_DATA:") == 1


def test_chunk_document_text_integrates_table_data():
    doc = (
        "# Town Animal Code\n\n"
        "## Sec. 6-132 Penalties\n\n"
        "| Offense | Fine |\n"
        "| :--- | :--- |\n"
        "| 1st offense | $20.00 |\n"
        "| 2nd offense | $50.00 |\n"
        "| 3rd offense | $75.00 |\n"
    )
    chunks = chunk_document_text("Town Code", doc)
    assert len(chunks) >= 1
    table_chunk = [c for c in chunks if "6-132" in c.section_title or "<!-- TABLE_DATA:" in c.content][0]
    assert table_chunk.chunk_type == "table"
    assert "<!-- TABLE_DATA:" in table_chunk.content
    assert '"3rd offense": "$75.00"' in table_chunk.content


def test_linearized_key_values_two_column():
    text = (
        "| Violation | Amount |\n"
        "| :--- | :--- |\n"
        "| Unrestrained dog | $50.00 |\n"
        "| Barking dog | $25.00 |\n"
    )
    res = serialize_tabular_data(text)
    assert "Key-Value Schedule:" in res
    assert "- **Unrestrained dog**: $50.00" in res
    assert "- **Barking dog**: $25.00" in res


def test_linearized_key_values_multi_column_omitted_for_lean_vectors():
    text = (
        "| Step | Department | Fee |\n"
        "| :--- | :--- | :--- |\n"
        "| Step 1 | Planning | $100 |\n"
        "| Step 2 | Engineering | $250 |\n"
    )
    res = serialize_tabular_data(text)
    assert "Key-Value Extraction:" not in res
    assert "| Step 1 | Planning | $100 |" in res


def test_linearized_key_values_plaintext_schedule():
    text = (
        "Offense Penalty\n"
        "1st     $ 20.00 fine\n"
        "2nd     $ 50.00 fine\n"
        "3rd     $ 75.00 fine\n"
    )
    res = serialize_tabular_data(text)
    assert "Key-Value Schedule:" in res
    assert "- **1st**: $ 20.00 fine" in res
    assert "- **2nd**: $ 50.00 fine" in res
    assert "- **3rd**: $ 75.00 fine" in res


def test_table_atomicity_pre_flush():
    prose = "This is a lengthy regulatory introduction explaining the statutory framework. " * 8
    table = (
        "| Offense | Penalty |\n"
        "| :--- | :--- |\n"
        "| Level 1 | $100 |\n"
        "| Level 2 | $200 |\n"
        "| Level 3 | $300 |\n"
    )
    doc = f"# Enforcement\n\n## Section 1\n\n{prose}\n\n{table}"
    chunks = chunk_document_text("Doc", doc, target_size=700, max_size=1200)

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    tc = table_chunks[0]
    assert "Level 1" in tc.content
    assert "Level 2" in tc.content
    assert "Level 3" in tc.content


def test_regression_cary_chicken_penalty_intact():
    prose = (
        "# Town Animal Code\n\n"
        "## Sec. 6-5 Backyard Chickens\n\n"
        "(a) Maximum number of female chickens (hens) allowed per lot is 5. Roosters are strictly prohibited.\n"
        "(b) Chicken coops must have a minimum setback of 15 feet from all property lines.\n\n"
        "## Sec. 6-132 Penalties\n\n"
        "To encourage responsible conduct, an owner shall be subject to escalating penalties for each violation of this section:\n\n"
        "Offense Penalty\n"
        "1st     $ 20.00 fine\n"
        "2nd     $ 50.00 fine\n"
        "3rd     $ 75.00 fine\n"
        "4th     $ 100.00 fine\n"
        "5th     $ 250.00 fine and seizure of the animal\n"
    )
    chunks = chunk_document_text("Cary Code", prose, target_size=1500, max_size=3000)
    penalty_chunks = [c for c in chunks if "6-132" in c.section_title or "Offense Penalty" in c.content]
    assert len(penalty_chunks) == 1
    c = penalty_chunks[0]
    for tier in ["1st", "2nd", "3rd", "4th", "5th"]:
        assert tier in c.content
    assert "$ 250.00 fine and seizure of the animal" in c.content
    assert "$ 50.00 fine" in c.content


def test_table_dimension_badge():
    text = (
        "| Parameter | Type | Default |\n"
        "| :--- | :--- | :--- |\n"
        "| alpha | float | 0.1 |\n"
        "| beta | int | 42 |\n"
    )
    res = serialize_tabular_data(text)
    assert "> **Table**: 2 rows × 3 columns" in res
    assert "| alpha | float | 0.1 |" in res


def test_section_overflow_preserves_table_atomicity():
    long_preamble = "Paragraph describing regulatory requirements and obligations under statutory mandates.\n\n" * 15
    table = (
        "Offense Penalty\n"
        "1st     $ 20.00 fine\n"
        "2nd     $ 50.00 fine\n"
        "3rd     $ 75.00 fine\n"
        "4th     $ 100.00 fine\n"
        "5th     $ 250.00 fine and seizure of the animal\n"
    )
    long_post = "\n\nAdditional enforcement measures may include injunctive relief or misdemeanor charges.\n\n" * 10
    doc = f"# Title\n\n## Section 1\n\n{long_preamble}\n\n{table}\n\n{long_post}"

    chunks = chunk_document_text("Test", doc, target_size=600, max_size=1200)
    t_chunks = [c for c in chunks if "1st" in c.content and "$ 20.00 fine" in c.content]
    assert len(t_chunks) == 1
    tc = t_chunks[0]
    for tier in ["1st", "2nd", "3rd", "4th", "5th"]:
        assert tier in tc.content
    assert "$ 250.00 fine and seizure of the animal" in tc.content


def test_massive_table_slicing_with_context():
    rows = [f"| Item {i} | Category {i % 3} | ${i * 10} |" for i in range(1, 31)]
    table_text = "| Item | Category | Price |\n| :--- | :--- | :--- |\n" + "\n".join(rows)
    doc = f"# Inventory\n\n## Catalog\n\n{table_text}"

    chunks = chunk_document_text("Store", doc, target_size=400, max_size=700)
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) > 1
    for c in table_chunks:
        assert "> **Table Context**: [Table: 30 rows × 3 columns | Showing rows" in c.content
        assert "| Item | Category | Price |" in c.content


def test_spec030_table_metadata_preservation():
    doc = (
        "# Finance\n\n## Revenue Breakdown\n\n"
        "| Product | 2025 | 2026 | Growth |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| iPhone | $200B | $220B | +10% |\n"
        "| Mac | $30B | $32B | +6.7% |\n"
    )
    chunks = chunk_document_text("FinanceDoc", doc)
    t_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(t_chunks) == 1
    tc = t_chunks[0]
    assert tc.chunk_type == "table"
    assert tc.table_headers == ["Product", "2025", "2026", "Growth"]
    assert tc.table_rows == 2
    assert tc.table_cols == 4


def test_spec030_schedule_metadata_preservation():
    doc = (
        "# Town Code\n\n## Section 6-132 Penalties\n\n"
        "Offense Penalty\n"
        "1st     $ 20.00 fine\n"
        "2nd     $ 50.00 fine\n"
        "3rd     $ 75.00 fine\n"
    )
    chunks = chunk_document_text("CaryCode", doc)
    s_chunks = [c for c in chunks if c.chunk_type in ("table", "schedule")]
    assert len(s_chunks) == 1
    sc = s_chunks[0]
    assert sc.chunk_type in ("table", "schedule")
    assert sc.table_headers == ["Offense", "Penalty"]
    assert sc.table_rows == 3
    assert sc.table_cols == 2
