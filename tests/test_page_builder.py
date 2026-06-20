from topic_overviews.kg.pagedata import PaperEntry, TopicPageData
from topic_overviews.wiki.page_builder import (
    build_topic_page, build_index_page, TOPIC_PAGE_PREFIX, INDEX_PAGE_TITLE,
)

DATA = TopicPageData(
    qid="Q11", label="Online Algorithms", description="Algorithms that act online.",
    papers=[
        PaperEntry(title="Online Caching", authors=["Jane Doe", "John Smith"],
                   year="2024", arxiv_id="2401.00001"),
    ],
)


def test_build_topic_page_exact():
    assert build_topic_page(DATA) == (
        "= Online Algorithms =\n"
        "\n"
        "Algorithms that act online.\n"
        "\n"
        '{| class="wikitable sortable"\n'
        "! Title !! Authors !! Year !! arXiv\n"
        "|-\n"
        "| Online Caching || Jane Doe; John Smith || 2024 || "
        "[https://arxiv.org/abs/2401.00001 2401.00001]\n"
        "|}\n"
    )


def test_build_index_page_exact():
    assert build_index_page([DATA]) == (
        "= Topic overview =\n"
        "\n"
        "* [[Topic:Online Algorithms|Online Algorithms]] (1 papers)\n"
    )


def test_constants():
    assert TOPIC_PAGE_PREFIX == "Topic:"
    assert INDEX_PAGE_TITLE == "Topic overview"
