"""Wiki pages the pipeline manages for research themes.

Theme pages render **live** via ``Template:ResearchTheme`` (which queries the
KG over SPARQL through ``Module:ResearchThemePublications``), so the pipeline
does not build paper tables in Python. It only ensures each theme's page holds
the template stub.

A theme's wiki page lives in the main namespace, titled by the theme label —
the same convention as ``Person`` profile pages (e.g. ``Erion Hasanbelliu``).
"""
from __future__ import annotations

# The entire content of a theme page: the template renders everything live.
RESEARCH_THEME_STUB = "{{ResearchTheme}}\n"
