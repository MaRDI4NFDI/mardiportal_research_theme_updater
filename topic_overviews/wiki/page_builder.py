"""Wiki pages the pipeline manages for research themes.

Theme pages render **live** via ``Template:ResearchTheme`` (which queries the
KG over SPARQL through ``Module:ResearchThemePublications``), so the pipeline
does not build paper tables in Python. It only ensures each theme's page holds
the template stub, and maintains a generated master index linking the theme
pages.

A theme's wiki page lives in the main namespace, titled by the theme label —
the same convention as ``Person`` profile pages (e.g. ``Erion Hasanbelliu``).
"""
from __future__ import annotations

from ..kg.topics import Topic

# The entire content of a theme page: the template renders everything live.
RESEARCH_THEME_STUB = "{{ResearchTheme}}\n"

# Title of the master index page listing all research themes.
INDEX_PAGE_TITLE = "Research themes"


def build_index_page(themes: list[Topic]) -> str:
    lines = [f"= {INDEX_PAGE_TITLE} =", ""]
    for t in themes:
        lines.append(f"* [[{t.label}]]")
    return "\n".join(lines) + "\n"
