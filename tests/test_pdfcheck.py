"""The comparison of a generated table with the printed one reports what is missing and finds what is there."""
from quantumrl_dissection.analysis import pdfcheck

PAGES = ["intro\n", "Table 9.9: A table.\nrow one   1.5   2.25\nrow two   10,934   0.5\n", "next page\n"]


def test_numbers_are_found_in_the_page_of_the_caption():
    res = pdfcheck.check_table(PAGES, "Table 9.9", [["row one", "1.5", "2.25"], ["row two", "10,934", "0.5"]], label_cols=1)
    assert res["found"] and res["checked"] == 4 and res["missing"] == []


def test_a_number_that_is_not_printed_is_reported():
    res = pdfcheck.check_table(PAGES, "Table 9.9", [["row one", "1.5", "7.75"]], label_cols=1)
    assert len(res["missing"]) == 1 and "7.75" in str(res["missing"][0])


def test_a_missing_caption_is_reported_not_raised():
    assert pdfcheck.check_table(PAGES, "Table 1.1", [["x", "1"]])["found"] is False
