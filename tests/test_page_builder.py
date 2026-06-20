from topic_overviews.kg.topics import Topic
from topic_overviews.wiki.page_builder import (
    build_index_page, RESEARCH_THEME_STUB, INDEX_PAGE_TITLE,
)

THEMES = [
    Topic(qid="Q11", label="Online Algorithms", description="..."),
    Topic(qid="Q12", label="Numerical Analysis", description="..."),
]


def test_research_theme_stub_is_just_the_template():
    assert RESEARCH_THEME_STUB == "{{ResearchTheme}}\n"


def test_index_page_title():
    assert INDEX_PAGE_TITLE == "Research themes"


def test_build_index_page_exact():
    assert build_index_page(THEMES) == (
        "= Research themes =\n"
        "\n"
        "* [[Online Algorithms]]\n"
        "* [[Numerical Analysis]]\n"
    )
