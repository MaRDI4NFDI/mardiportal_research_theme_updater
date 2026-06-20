from topic_overviews.kg.topics import Topic
from topic_overviews.kg.pagedata import PaperEntry, TopicPageData, fetch_topic_page_data

TOPIC = Topic(qid="Q11", label="Online Algorithms", description="Algorithms online.")


def test_fetch_topic_page_data_builds_entries():
    captured = {}

    def fake_run(endpoint, query):
        captured["query"] = query
        return [
            {"title": "Online Caching", "year": "2024-01-02",
             "arxiv": "2401.00001", "authors": "Jane Doe; John Smith"},
            {"title": "Ski Rental Revisited", "year": "2023-11-09",
             "arxiv": "2311.00050", "authors": "Ada Lovelace"},
        ]

    data = fetch_topic_page_data("http://ep", TOPIC, run=fake_run)
    assert data == TopicPageData(
        qid="Q11", label="Online Algorithms", description="Algorithms online.",
        papers=[
            PaperEntry(title="Online Caching", authors=["Jane Doe", "John Smith"],
                       year="2024", arxiv_id="2401.00001"),
            PaperEntry(title="Ski Rental Revisited", authors=["Ada Lovelace"],
                       year="2023", arxiv_id="2311.00050"),
        ],
    )
    assert "Q11" in captured["query"]
