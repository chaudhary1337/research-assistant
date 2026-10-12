"""
Research Assistant (RA) Slack app. Unfurls arXiv links with the full title, authors, and abstract.
With --tabs on, the card gets one button for the abstract and one per top-level section whose heading
contains any of --section-keywords, labeled with that heading. Clicking switches the card text for everyone.
Headings always lose their section number; --heading-style title also normalizes their case to Title Case.
With --subheadings bold, subsection and paragraph headings inside a tab appear as bold lines without numbers.
Math is shown as Unicode (--math-style unicode, e.g. ±180^∘) or as its LaTeX source (--math-style latex).
Sections come from arXiv's HTML rendering; with --pdf-fallback on, the PDF is used when the HTML has none.
"""
import argparse
import functools
import itertools
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass

import httpx
import pymupdf
import pymupdf4llm
from bs4 import BeautifulSoup, NavigableString, Tag
from loguru import logger
from pylatexenc.latex2text import LatexNodes2Text
from slack_bolt import Ack, App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient

ARXIV_ID_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(v\d+)?"
)
TAB_ACTION_PATTERN = re.compile(r"^show_tab_\d+$")
HEADING_NUMBER_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+")
ABS_URL = "https://arxiv.org/abs/{paper_id}"
HTML_URL = "https://arxiv.org/html/{paper_id}"
PDF_URL = "https://arxiv.org/pdf/{paper_id}"
HTTP_HEADERS = {"User-Agent": "research-assistant-slack-bot (+https://github.com/chaudhary1337/research-assistant)"}
HTTP_TIMEOUT_SECONDS = 15
PDF_TIMEOUT_SECONDS = 120
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#+)\s+(.*?)\s*$")
MARKDOWN_RUN_IN_PATTERN = re.compile(r"^\*\*([^*]+?[.:])\*\*\s*(.*)$", re.DOTALL)
SECTION_CACHE_SIZE = 128
MAX_AUTHORS = 8
SLACK_TEXT_LIMIT = 3000
SLACK_BUTTON_TEXT_LIMIT = 75
MAX_BODY_BLOCKS = 45
ABSTRACT_TAB = "Abstract"
TITLE_CASE_SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on", "or", "the", "to", "via", "vs", "with"}


@dataclass(frozen=True)
class Settings:
    tabs_enabled: bool
    keywords: tuple[str, ...]
    heading_style: str
    math_style: str
    subheadings: str
    pdf_fallback: bool


Part = tuple[bool, str]


latex_converter = LatexNodes2Text(math_mode="text", keep_braced_groups=True, keep_braced_groups_minlen=2)


@dataclass
class Paper:
    title: str
    authors: list[str]
    date: str
    abstract: str


def escape(text: str) -> str:
    """
    Escape the three characters Slack mrkdwn treats as control characters.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def get_page(url: str) -> BeautifulSoup | None:
    """
    Download and parse a page, returning None when it does not exist.
    """
    response = httpx.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def fetch_paper(paper_id: str) -> Paper:
    """
    Read a paper's metadata from the citation meta tags on its arXiv abstract page.
    The export API is avoided because it rate-limits aggressively (HTTP 429).
    """
    soup = get_page(ABS_URL.format(paper_id=paper_id))

    def meta(name: str) -> list[str]:
        return [tag["content"] for tag in soup.find_all("meta", attrs={"name": name})]

    return Paper(
        title=meta("citation_title")[0],
        authors=[" ".join(reversed(name.split(", ", 1))) for name in meta("citation_author")],
        date=meta("citation_date")[0].replace("/", "-"),
        abstract=" ".join(meta("citation_abstract")[0].split()),
    )


def normalize_heading(raw: str, heading_style: str) -> str:
    """
    Collapse whitespace and drop a leading section number ("5", "5.1", "V."). With heading_style "title",
    also convert to Title Case: all-caps headings are lowercased first, lowercase words are capitalized
    except small words after the first, and words with capitals (acronyms, names) are kept as written.
    """
    heading = HEADING_NUMBER_PATTERN.sub("", " ".join(raw.split()))
    if heading_style != "title":
        return heading
    if heading.isupper():
        heading = heading.lower()
    words = heading.split(" ")
    return " ".join(
        word if not word.islower() or (index > 0 and word in TITLE_CASE_SMALL_WORDS) else word[0].upper() + word[1:]
        for index, word in enumerate(words)
    )


def run_in_heading(paragraph: Tag) -> Tag | None:
    """
    Return the bold lead-in that opens a paragraph ("Setup. We train ...") when it ends with "." or ":".
    """
    first = next((child for child in paragraph.children if not isinstance(child, NavigableString) or child.strip()), None)
    if isinstance(first, Tag) and "ltx_font_bold" in first.get("class", []) and first.get_text().strip().endswith((".", ":")):
        return first
    return None


def subheading(raw: str, heading_style: str) -> str:
    """
    Normalize a subheading like a section heading and drop its trailing "." or ":".
    """
    return normalize_heading(raw, heading_style).rstrip(".:")


def keyword_pattern(settings: Settings) -> re.Pattern:
    """
    Compile the section keywords into one case-insensitive pattern.
    """
    return re.compile("|".join(map(re.escape, settings.keywords)), re.IGNORECASE)


def fetch_html_sections(paper_id: str, heading_pattern: re.Pattern, settings: Settings) -> list[tuple[str, list[Part]]]:
    """
    Return (heading, parts) for every top-level section whose heading matches heading_pattern, in paper order,
    read from arXiv's HTML rendering. Parts are (is_subheading, text) in document order; subheadings, including
    bold run-in paragraph leads, are kept only with settings.subheadings == "bold". All headings go through
    normalize_heading after arXiv's separate number tag is removed. Text inside figures and tables is skipped, and math follows settings.math_style.
    """
    soup = get_page(HTML_URL.format(paper_id=paper_id))
    if soup is None:
        return []
    sections = []
    for section in soup.select("section.ltx_section"):
        title = section.find(class_="ltx_title")
        if title is None:
            continue
        for number in section.select(".ltx_title .ltx_tag"):
            number.decompose()
        heading = normalize_heading(title.get_text(), settings.heading_style)
        if not heading_pattern.search(heading):
            continue
        for note in section.select(".ltx_note"):
            note.decompose()
        for math in section.find_all("math"):
            latex = math.get("alttext", "")
            math.replace_with(latex_converter.latex_to_text(latex) if settings.math_style == "unicode" else latex)
        parts = []
        for element in section.find_all(["p", "h2", "h3", "h4", "h5", "h6"]):
            text = " ".join(element.get_text().split())
            if element is title or not text or element.find_parent(["figure", "table"]) is not None:
                continue
            if element.name == "p":
                lead = run_in_heading(element) if settings.subheadings == "bold" else None
                if lead is not None:
                    parts.append((True, subheading(lead.get_text(), settings.heading_style)))
                    lead.decompose()
                    text = " ".join(element.get_text().split())
                if text:
                    parts.append((False, text))
            elif settings.subheadings == "bold" and "ltx_title" in element.get("class", []):
                parts.append((True, subheading(text, settings.heading_style)))
        if any(not is_subheading for is_subheading, _ in parts):
            sections.append((heading, parts))
    return sections


def clean_markdown(text: str) -> str:
    """
    Remove markdown emphasis that pymupdf4llm adds (bold and italic math such as "7 _._ 1 _×_"),
    rejoin decimals split by it, and collapse whitespace.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = " ".join(text.split())
    return re.sub(r"(\d) \. (\d)", r"\1.\2", text)


def fetch_pdf_sections(paper_id: str, heading_pattern: re.Pattern, settings: Settings) -> list[tuple[str, list[Part]]]:
    """
    Same output as fetch_html_sections, read from the PDF. Pages whose plain text has a line starting with a
    matching heading are found first, and only those pages and the next one are converted to markdown, since
    converting a whole paper takes tens of seconds. A section ends at the next heading of the same or a higher
    level with the same emphasis; other headings inside it are subheadings.
    """
    response = httpx.get(PDF_URL.format(paper_id=paper_id), headers=HTTP_HEADERS, timeout=PDF_TIMEOUT_SECONDS, follow_redirects=True)
    response.raise_for_status()
    document = pymupdf.open(stream=response.content, filetype="pdf")
    line_pattern = re.compile(rf"^\s*(?:[IVXLC]+|\d+)?\.?\s*(?:{heading_pattern.pattern})", re.IGNORECASE | re.MULTILINE)
    candidates = [index for index, page in enumerate(document) if line_pattern.search(page.get_text())]
    pages = sorted({page for index in candidates for page in (index, index + 1) if page < document.page_count})
    if not pages:
        return []
    markdown = pymupdf4llm.to_markdown(document, pages=pages, ignore_images=True, ignore_graphics=True, show_progress=False)
    sections = []
    current = None
    for chunk in re.split(r"\n\s*\n", markdown):
        chunk = chunk.strip()
        match = MARKDOWN_HEADING_PATTERN.match(chunk)
        if match is not None:
            level, emphasized = len(match.group(1)), match.group(2).startswith(("_", "*"))
            heading = normalize_heading(clean_markdown(match.group(2)), settings.heading_style)
            if current is not None and level >= current["level"] and not (level == current["level"] and emphasized == current["emphasized"]):
                if settings.subheadings == "bold":
                    current["parts"].append((True, heading.rstrip(".:")))
                continue
            if current is not None:
                sections.append((current["heading"], current["parts"]))
                current = None
            if heading_pattern.search(heading):
                current = {"heading": heading, "level": level, "emphasized": emphasized, "parts": []}
        elif current is not None and chunk:
            lead = MARKDOWN_RUN_IN_PATTERN.match(chunk) if settings.subheadings == "bold" else None
            if lead is not None:
                current["parts"].append((True, subheading(lead.group(1), settings.heading_style)))
                chunk = lead.group(2)
            text = clean_markdown(chunk)
            if text:
                current["parts"].append((False, text))
    if current is not None:
        sections.append((current["heading"], current["parts"]))
    return [(heading, parts) for heading, parts in sections if any(not is_subheading for is_subheading, _ in parts)]


@functools.lru_cache(maxsize=SECTION_CACHE_SIZE)
def paper_sections(paper_id: str, settings: Settings) -> list[tuple[str, list[Part]]]:
    """
    Matching sections from the HTML, falling back to the PDF when the HTML has none and settings.pdf_fallback is on.
    Cached per paper so button clicks do not download the paper again; failed fetches raise and are not cached.
    """
    heading_pattern = keyword_pattern(settings)
    sections = fetch_html_sections(paper_id, heading_pattern, settings)
    if not sections and settings.pdf_fallback:
        logger.info(f"No matching HTML sections for {paper_id}, reading the PDF")
        sections = fetch_pdf_sections(paper_id, heading_pattern, settings)
    return sections


def split_text(text: str) -> list[str]:
    """
    Pack paragraphs into chunks of at most SLACK_TEXT_LIMIT characters, never cutting inside a paragraph
    unless it alone is too long, in which case it is cut at word boundaries.
    """
    chunks = []
    for paragraph in text.split("\n\n"):
        for piece in textwrap.wrap(paragraph, SLACK_TEXT_LIMIT, break_long_words=False) or [""]:
            if chunks and len(chunks[-1]) + 2 + len(piece) <= SLACK_TEXT_LIMIT:
                chunks[-1] += "\n\n" + piece
            else:
                chunks.append(piece)
    return chunks


def build_unfurl(paper: Paper, url: str, tabs: list[tuple[str, list[Part]]], paper_id: str, active_tab: int) -> dict:
    """
    Build Slack blocks: title, authors, and date, then the active tab's parts (subheadings as bold mrkdwn,
    paragraph runs as plain text), then one button per tab when there is more than one tab.
    """
    authors = ", ".join(paper.authors[:MAX_AUTHORS]) + (" et al." if len(paper.authors) > MAX_AUTHORS else "")
    header = f"*<{url}|{escape(paper.title)}>*\n{escape(authors)} · {paper.date}"
    body_blocks = []
    for is_subheading, group in itertools.groupby(tabs[active_tab][1], key=lambda part: part[0]):
        texts = [text for _, text in group]
        if is_subheading:
            body_blocks += [{"type": "section", "text": {"type": "mrkdwn", "text": f"*{escape(text)}*"}} for text in texts]
        else:
            body_blocks += [{"type": "section", "text": {"type": "plain_text", "text": chunk}} for chunk in split_text("\n\n".join(texts))]
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": header}}] + body_blocks[:MAX_BODY_BLOCKS]
    if len(tabs) > 1:
        buttons = []
        for index, (label, _) in enumerate(tabs):
            button = {
                "type": "button",
                "action_id": f"show_tab_{index}",
                "text": {"type": "plain_text", "text": label[:SLACK_BUTTON_TEXT_LIMIT]},
                "value": json.dumps({"paper_id": paper_id, "tab": index}),
            }
            if index == active_tab:
                button["style"] = "primary"
            buttons.append(button)
        blocks.append({"type": "actions", "elements": buttons})
    return {"blocks": blocks}


def render(paper_id: str, url: str, active_tab: int, settings: Settings) -> dict:
    """
    Fetch the paper and its matching sections, and build its card showing the requested tab.
    """
    paper = fetch_paper(paper_id)
    tabs = [(ABSTRACT_TAB, [(False, paper.abstract)])]
    if settings.tabs_enabled:
        try:
            tabs += paper_sections(paper_id, settings)
        except (httpx.HTTPError, pymupdf.FileDataError) as error:
            logger.warning(f"Could not read sections for {paper_id}: {error!r}")
    if active_tab >= len(tabs):
        active_tab = 0
    return build_unfurl(paper, url, tabs, paper_id, active_tab)


def create_app(token: str, settings: Settings) -> App:
    """
    Create the Bolt app with the link and button listeners.
    """
    app = App(token=token)

    @app.event("link_shared")
    def handle_link_shared(event: dict, client: WebClient) -> None:
        """
        Replace Slack's default preview of each arXiv link with our own preview card.
        """
        urls = [link["url"] for link in event["links"]]
        logger.info(f"link_shared source={event.get('source')} channel={event.get('channel')} bot_member={event.get('is_bot_user_member')} urls={urls}")
        if event.get("source") == "composer":
            return
        unfurls = {}
        for link in event["links"]:
            match = ARXIV_ID_PATTERN.search(link["url"])
            if match is None:
                logger.warning(f"No arXiv id in {link['url']}")
                continue
            paper_id = match.group(1) + (match.group(2) or "")
            try:
                unfurls[link["url"]] = render(paper_id, link["url"], 0, settings)
            except (httpx.HTTPError, IndexError, TypeError) as error:
                logger.warning(f"Could not read arXiv page for {paper_id}: {error!r}")
        if not unfurls:
            return
        client.chat_unfurl(channel=event["channel"], ts=event["message_ts"], unfurls=unfurls)
        logger.info(f"Unfurled {list(unfurls)}")

    @app.action(TAB_ACTION_PATTERN)
    def handle_tab_click(ack: Ack, body: dict, client: WebClient) -> None:
        """
        Switch an existing card to the clicked tab by unfurling the same link again.
        """
        ack()
        value = json.loads(body["actions"][0]["value"])
        container = body["container"]
        url = container["app_unfurl_url"]
        logger.info(f"tab_click user={body['user']['id']} paper={value['paper_id']} tab={value['tab']}")
        unfurl = render(value["paper_id"], url, value["tab"], settings)
        client.chat_unfurl(channel=container["channel_id"], ts=container["message_ts"], unfurls={url: unfurl})
        logger.info(f"Switched {url} to tab {value['tab']}")

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Assistant (RA) Slack bot.")
    parser.add_argument(
        "--tabs",
        choices=["on", "off"],
        required=True,
        help="on: Abstract button plus one button per matching section, switching the card text for everyone; off: abstract only",
    )
    parser.add_argument(
        "--section-keywords",
        nargs="+",
        required=True,
        help="case-insensitive words; every top-level section whose heading contains any of them gets its own button",
    )
    parser.add_argument(
        "--heading-style",
        choices=["title", "as-written"],
        required=True,
        help="title: button labels in Title Case (V CONCLUSION -> Conclusion); as-written: keep the paper's case. Numbers are always dropped",
    )
    parser.add_argument(
        "--math-style",
        choices=["unicode", "latex"],
        required=True,
        help="unicode: convert math to readable symbols (\\pm 180^{\\circ} -> ±180^∘); latex: show the LaTeX source",
    )
    parser.add_argument(
        "--subheadings",
        choices=["bold", "omit"],
        required=True,
        help="bold: show subsection and paragraph headings inside a tab as bold lines without numbers; omit: text only",
    )
    parser.add_argument(
        "--pdf-fallback",
        choices=["on", "off"],
        required=True,
        help="on: read sections from the PDF when arXiv has no HTML version or the HTML has no matching section; off: HTML only",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = Settings(
        tabs_enabled=args.tabs == "on",
        keywords=tuple(args.section_keywords),
        heading_style=args.heading_style,
        math_style=args.math_style,
        subheadings=args.subheadings,
        pdf_fallback=args.pdf_fallback == "on",
    )
    app = create_app(os.environ["SLACK_BOT_TOKEN"], settings)
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
