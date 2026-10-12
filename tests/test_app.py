"""
Tests for the RA bot. Offline tests cover the text helpers; tests marked network fetch real papers from arXiv
and pin the buttons and text each paper posted in Slack so far should produce. Run all with `uv run pytest`,
or offline only with `uv run pytest -m "not network"`.
"""
import pytest
from bs4 import BeautifulSoup

from app import SLACK_TEXT_LIMIT, Settings, clean_markdown, normalize_heading, paper_sections, render, run_in_heading, split_text

SETTINGS = Settings(
    tabs_enabled=True,
    keywords=("conclusion", "discussion"),
    heading_style="title",
    math_style="unicode",
    subheadings="bold",
    pdf_fallback=True,
)

PAPER_BUTTONS = {
    "2306.11565": ["Abstract", "Conclusions and Future Work"],
    "2309.13037": ["Abstract", "Discussion"],
    "2312.02976": ["Abstract", "Conclusion"],
    "2509.16615": ["Abstract", "Conclusion and Discussion"],
    "2511.10110": ["Abstract", "Discussion"],
    "2603.00615": ["Abstract", "Conclusion"],
    "2606.04233": ["Abstract", "Discussion", "Conclusion"],
    "2607.27203": ["Abstract", "Discussion"],
    "2609.36720v1": ["Abstract", "Conclusion"],
    "2610.03604": ["Abstract", "Conclusion"],
    "2610.09254": ["Abstract", "Conclusion and Limitations"],
    "2610.10515": ["Abstract", "Conclusion"],
}


@pytest.mark.parametrize(
    "raw, title, as_written",
    [
        ("V CONCLUSION", "Conclusion", "CONCLUSION"),
        ("VII Conclusion and Discussion", "Conclusion and Discussion", "Conclusion and Discussion"),
        ("5  Conclusion", "Conclusion", "Conclusion"),
        ("6 Conclusions and Future Work", "Conclusions and Future Work", "Conclusions and Future Work"),
        ("VI. CONCLUSION AND LIMITATIONS", "Conclusion and Limitations", "CONCLUSION AND LIMITATIONS"),
        ("5.1 Discussion", "Discussion", "Discussion"),
        ("IPE: Initialization via Policy Ensemble", "IPE: Initialization via Policy Ensemble", "IPE: Initialization via Policy Ensemble"),
        ("3  T²Mem", "T²Mem", "T²Mem"),
    ],
)
def test_normalize_heading(raw: str, title: str, as_written: str) -> None:
    """
    Numbers are always dropped; only the title style changes case.
    """
    assert normalize_heading(raw, "title") == title
    assert normalize_heading(raw, "as-written") == as_written


def test_split_text_keeps_words_and_paragraphs_whole() -> None:
    """
    Chunks stay under Slack's limit, break between paragraphs, and never cut a word.
    """
    paragraphs = [f"paragraph{index} " + "word " * 400 for index in range(5)]
    chunks = split_text("\n\n".join(paragraph.strip() for paragraph in paragraphs))
    assert all(len(chunk) <= SLACK_TEXT_LIMIT for chunk in chunks)
    assert all(chunk.startswith("paragraph") for chunk in chunks)
    assert {word for chunk in chunks for word in chunk.split()} <= {"word"} | {f"paragraph{index}" for index in range(5)}


def test_split_text_cuts_long_paragraph_at_spaces() -> None:
    """
    A single paragraph longer than the limit is cut only at spaces.
    """
    chunks = split_text("word " * 2000)
    assert len(chunks) > 1
    assert all(set(chunk.split()) == {"word"} for chunk in chunks)


def test_clean_markdown_rejoins_italic_math() -> None:
    """
    pymupdf4llm's italic markers around math are removed and split decimals are rejoined.
    """
    assert clean_markdown("about 7 _._ 1 _×_ and 3 _._ 6 _×_ the rate") == "about 7.1 × and 3.6 × the rate"
    assert clean_markdown("**Limitations.** text with x_t kept") == "Limitations. text with x_t kept"


def test_run_in_heading_needs_leading_bold_with_period() -> None:
    """
    Only a bold phrase that opens the paragraph and ends with "." or ":" counts as a subheading.
    """
    lead = BeautifulSoup('<p><span class="ltx_font_bold">Setup.</span> We train.</p>', "html.parser").p
    middle = BeautifulSoup('<p>We <span class="ltx_font_bold">really</span> train.</p>', "html.parser").p
    no_period = BeautifulSoup('<p><span class="ltx_font_bold">Note</span> that we train.</p>', "html.parser").p
    assert run_in_heading(lead).get_text() == "Setup."
    assert run_in_heading(middle) is None
    assert run_in_heading(no_period) is None


@pytest.mark.network
@pytest.mark.parametrize("paper_id, buttons", PAPER_BUTTONS.items())
def test_paper_buttons_and_text(paper_id: str, buttons: list[str]) -> None:
    """
    Each paper posted so far gets the expected buttons, and every tab has text within Slack's limits.
    """
    first = render(paper_id, f"https://arxiv.org/abs/{paper_id}", 0, SETTINGS)["blocks"]
    assert [button["text"]["text"] for button in first[-1]["elements"]] == buttons
    for tab in range(len(buttons)):
        body = render(paper_id, f"https://arxiv.org/abs/{paper_id}", tab, SETTINGS)["blocks"][1:-1]
        texts = [block["text"]["text"] for block in body if block["text"]["type"] == "plain_text"]
        assert texts and all(0 < len(text) <= SLACK_TEXT_LIMIT for text in texts)


@pytest.mark.network
def test_subheadings_and_math() -> None:
    """
    Subsection titles and bold run-in leads become subheadings, and math is converted to Unicode.
    """
    thousand_tasks = dict(paper_sections("2511.10110", SETTINGS))["Discussion"]
    assert [text for is_subheading, text in thousand_tasks if is_subheading] == [
        "Learning From Limited Per-task Demonstrations",
        "Data Requirements and Scaling Properties of Behavioural Cloning",
        "Limitations and Scope of the Current Study",
    ]
    assert any("±180" in text for _, text in thousand_tasks)
    benchmarking = dict(paper_sections("2606.04233", SETTINGS))["Discussion"]
    assert "Until Such Benchmarks Exist" in [text for is_subheading, text in benchmarking if is_subheading]


@pytest.mark.network
def test_pdf_fallback() -> None:
    """
    A paper without an HTML version gets its sections from the PDF, with clean math and a run-in subheading.
    """
    parts = dict(paper_sections("2610.09254", SETTINGS))["Conclusion and Limitations"]
    texts = [text for is_subheading, text in parts if not is_subheading]
    assert any("7.1 ×" in text for text in texts)
    assert (True, "Limitations") in parts
    without_fallback = Settings(**{**SETTINGS.__dict__, "pdf_fallback": False})
    assert paper_sections("2610.09254", without_fallback) == []
