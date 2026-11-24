import pytest

from maintenance.dataset_importer import extract_doi


@pytest.mark.parametrize("uid, expected", [
    ("https://doi.org/10.5281/zenodo.123", "10.5281/ZENODO.123"),
    ("http://dx.doi.org/10.1000/abc", "10.1000/ABC"),
    ("HTTPS://DOI.ORG/10.1000/abc", "10.1000/ABC"),
    ("doi:10.1000/abc", "10.1000/ABC"),
    ("10.1000/abc", "10.1000/ABC"),
    ("  https://doi.org/10.1000/abc \n", "10.1000/ABC"),
])
def test_doi_is_bare_and_upper_case(uid, expected):
    assert extract_doi(uid) == expected


@pytest.mark.parametrize("uid", [None, "", "https://example.org/x", "zenodo.123"])
def test_non_doi_gives_none(uid):
    assert extract_doi(uid) is None
