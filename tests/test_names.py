from topic_overviews.kg.names import normalize_author_name, normalize_orcid


def test_normalize_author_name():
    assert normalize_author_name("Doe, Jane") == "Jane Doe"
    assert normalize_author_name("van der Berg, Anna Maria") == "Anna Maria van der Berg"
    assert normalize_author_name("Jane Doe") == "Jane Doe"
    assert normalize_author_name("Aristotle,") == "Aristotle"


def test_normalize_orcid():
    assert normalize_orcid("https://orcid.org/0000-0002-1825-0097") == "0000-0002-1825-0097"
    assert normalize_orcid("http://orcid.org/0000-0002-1825-0097/") == "0000-0002-1825-0097"
    assert normalize_orcid("0000-0002-1825-0097") == "0000-0002-1825-0097"
    assert normalize_orcid("0000-0002-1694-233x") == "0000-0002-1694-233X"
    assert normalize_orcid("") == ""
    assert normalize_orcid(None) == ""
    assert normalize_orcid("not-an-orcid") == ""
    assert normalize_orcid("https://orcid.org/0000-0002-1825") == ""
