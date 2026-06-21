from topic_overviews.kg.topics import Topic, load_registered_topics


def test_load_registered_topics_parses_rows():
    captured = {}

    def fake_run(endpoint, query):
        captured["endpoint"] = endpoint
        captured["query"] = query
        return [
            {"topic": "https://x/entity/Q10", "label": "Optimization",
             "desc": "Mathematical optimization."},
            {"topic": "https://x/entity/Q11", "label": "Numerical Analysis"},
        ]

    topics = load_registered_topics("http://ep", "Q5", run=fake_run)
    assert topics == [
        Topic(qid="Q10", label="Optimization", description="Mathematical optimization."),
        Topic(qid="Q11", label="Numerical Analysis", description=""),
    ]
    assert "Q5" in captured["query"]            # filters on the overview-topic class
    assert captured["endpoint"] == "http://ep"


def test_load_registered_topics_can_read_arxiv_query_property():
    captured = {}

    def fake_run(endpoint, query):
        captured["query"] = query
        return [
            {
                "topic": "https://x/entity/Q11",
                "label": "Online Algorithms",
                "arxivQuery": 'cat:cs.DS OR all:"online algorithms"',
            },
        ]

    topics = load_registered_topics(
        "http://ep",
        "Q5",
        arxiv_query_property="P999",
        run=fake_run,
    )
    assert topics == [
        Topic(
            qid="Q11",
            label="Online Algorithms",
            description="",
            arxiv_query='cat:cs.DS OR all:"online algorithms"',
        ),
    ]
    assert "wdt:P999" in captured["query"]
    assert "?arxivQuery" in captured["query"]


def test_load_registered_topics_empty():
    assert load_registered_topics("http://ep", "Q5", run=lambda e, q: []) == []
