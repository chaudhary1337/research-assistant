# Research Assistant (RA)

Slack bot that turns arXiv links into cards with the title, authors, and full abstract, plus buttons for the paper's Conclusion / Discussion sections. Sections come from arXiv's HTML, falling back to the PDF.

## Setup

1. Create a Slack app from `manifest.yml` (api.slack.com/apps → Create New App → From a manifest) and install it.
2. Export the Bot User OAuth Token and an App-Level Token with `connections:write`:
   ```bash
   export SLACK_BOT_TOKEN=xoxb-... SLACK_APP_TOKEN=xapp-...
   ```
3. Run it. Every setting is required; `uv run app.py --help` explains each one.
   ```bash
   uv run --no-dev app.py --tabs on --section-keywords conclusion discussion --heading-style title --math-style unicode --subheadings bold --pdf-fallback on
   ```

If cards fail with `cannot_unfurl_url`, remove `arxiv.org` from Slack's Admin → Workspace settings → Attachments → Blocked previews.

## Tests

```bash
uv run pytest                  # all, including live arXiv fetches
uv run pytest -m "not network" # offline only
```

## License

AGPL-3.0-or-later (see `LICENSE`). The PDF fallback depends on PyMuPDF, which is licensed under AGPL-3.0, so this project uses the same license to stay compatible.
