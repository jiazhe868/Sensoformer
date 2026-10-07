#!/usr/bin/env python3
"""
The YHS focal-mechanism catalog is third-party data that users must download
themselves (docs/DATA_PIPELINE.md). Because a mis-ordered catalog would
silently attach the wrong mechanism and quality grade to every event, the
parser must fail loudly rather than guess.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "preprocessing"))
from preprocess_real_hdf5 import (CatalogFormatError, VALID_GRADES, YSH_COL,
                                  YSH_MIN_COLUMNS, parse_ysh_catalog)

GOOD = (
    " 1981  1  1  4 13 55.710  3301565  33.25517 -115.96750   5.680  2.260"
    "  318  57 -168  37  39   18  0.17    0  0.00 C \n"
    " 2004  4 15  2 28  8.620 10000605  33.94280 -116.99420  15.760  3.350"
    "  145  85  -12  20  21   45  0.08   12  1.00 A \n"
)


def write(tmp_path, text, name="cat.log"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_parses_expected_layout(tmp_path):
    ev = parse_ysh_catalog(write(tmp_path, GOOD))
    assert len(ev) == 2
    first, second = ev
    assert first["event_id"] == "3301565" and first["grade"] == "C"
    assert first["date"] == 19810101
    assert first["magnitude"] == pytest.approx(2.26)
    assert (first["strike"], first["dip"], first["rake"]) == (318.0, 57.0, -168.0)
    assert second["event_id"] == "10000605" and second["grade"] == "A"
    assert second["lat"] == pytest.approx(33.94280)
    assert second["lon"] == pytest.approx(-116.99420)


def test_comments_and_blank_lines_ignored(tmp_path):
    text = "# a header comment\n\n" + GOOD
    assert len(parse_ysh_catalog(write(tmp_path, text))) == 2


def test_too_few_columns_raises_with_guidance(tmp_path):
    truncated = "\n".join(" ".join(l.split()[:12]) for l in GOOD.strip().split("\n"))
    with pytest.raises(CatalogFormatError, match="at least 21"):
        parse_ysh_catalog(write(tmp_path, truncated))


def test_grade_column_not_a_grade_raises(tmp_path):
    """Shifted columns must be caught, not silently accepted."""
    shifted = GOOD.replace(" C \n", " 99 \n")
    with pytest.raises(CatalogFormatError, match="quality grade"):
        parse_ysh_catalog(write(tmp_path, shifted))


def test_empty_catalog_raises(tmp_path):
    with pytest.raises(CatalogFormatError, match="No usable rows"):
        parse_ysh_catalog(write(tmp_path, "# only a comment\n"))


def test_column_map_is_self_consistent():
    assert YSH_COL["quality"] == YSH_MIN_COLUMNS - 1
    assert set("ABCD") <= VALID_GRADES
    assert len(set(YSH_COL.values())) == len(YSH_COL)   # no duplicate indices
