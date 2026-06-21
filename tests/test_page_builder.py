from topic_overviews.wiki.page_builder import RESEARCH_THEME_STUB


def test_research_theme_stub_is_just_the_template():
    assert RESEARCH_THEME_STUB == "{{ResearchTheme}}\n"
