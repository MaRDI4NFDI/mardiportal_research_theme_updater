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


def test_load_registered_topics_empty():
    assert load_registered_topics("http://ep", "Q5", run=lambda e, q: []) == []
