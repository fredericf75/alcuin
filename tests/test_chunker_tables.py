"""
tests/test_chunker_tables.py - Unit tests for SPEC-007: Structural Tabular Serialization
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
    assert "<!-- TABLE_DATA:" in result
    assert '"Department": "Animal Control"' in result
    assert '"Days": "5"' in result
    assert '"Department": "Housing Inspector"' in result
    assert '"Days": "10"' in result


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
