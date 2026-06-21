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


def test_load_topics_includes_openalex_query():
    rows = [
        {
            "topic": "https://portal.mardi4nfdi.de/entity/Q7266564",
            "label": "MaRDI",
            "desc": "MaRDI theme",
            "arxivQuery": "all:mardi",
            "openalexQuery": "search=mardi&filter=funders.id:f4320320879",
        }
    ]
    topics = load_registered_topics(
        "http://sparql",
        "Q7266523",
        arxiv_query_property="P1965",
        openalex_query_property="P1967",
        run=lambda endpoint, query: rows,
    )
    assert topics[0].openalex_query == "search=mardi&filter=funders.id:f4320320879"


def test_load_topics_openalex_query_defaults_to_empty():
    rows = [
        {
            "topic": "https://portal.mardi4nfdi.de/entity/Q7266564",
            "label": "MaRDI",
            "desc": "",
            "arxivQuery": "",
        }
    ]
    topics = load_registered_topics(
        "http://sparql",
        "Q7266523",
        arxiv_query_property="P1965",
        openalex_query_property="P1967",
        run=lambda endpoint, query: rows,
    )
    assert topics[0].openalex_query == ""


def test_load_topics_omits_openalex_when_property_not_configured():
    rows = [
        {
            "topic": "https://portal.mardi4nfdi.de/entity/Q7266564",
            "label": "MaRDI",
            "desc": "",
            "arxivQuery": "",
        }
    ]
    captured = []
    load_registered_topics(
        "http://sparql",
        "Q7266523",
        arxiv_query_property="P1965",
        openalex_query_property="",
        run=lambda endpoint, query: captured.append(query) or rows,
    )
    assert "openalexQuery" not in captured[0]


def test_load_topics_includes_since_days():
    rows = [
        {
            "topic": "https://portal.mardi4nfdi.de/entity/Q7266564",
            "label": "MaRDI",
            "desc": "",
            "sinceDays": "30",
        }
    ]
    topics = load_registered_topics(
        "http://sparql",
        "Q7266523",
        since_days_property="P1968",
        run=lambda endpoint, query: rows,
    )
    assert topics[0].since_days == 30


def test_load_topics_since_days_defaults_to_none():
    rows = [{"topic": "https://portal.mardi4nfdi.de/entity/Q7266564", "label": "MaRDI", "desc": ""}]
    topics = load_registered_topics(
        "http://sparql",
        "Q7266523",
        since_days_property="P1968",
        run=lambda endpoint, query: rows,
    )
    assert topics[0].since_days is None


def test_load_topics_omits_since_days_when_property_not_configured():
    rows = [{"topic": "https://portal.mardi4nfdi.de/entity/Q7266564", "label": "MaRDI", "desc": ""}]
    captured = []
    load_registered_topics(
        "http://sparql",
        "Q7266523",
        since_days_property="",
        run=lambda endpoint, query: captured.append(query) or rows,
    )
    assert "sinceDays" not in captured[0]
    assert "P1968" not in captured[0]
