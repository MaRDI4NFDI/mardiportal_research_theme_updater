from unittest.mock import patch, MagicMock
from topic_overviews.arxiv_to_md import fetch_and_convert, _convert_html


MINIMAL_HTML = """
<html><body>
<p>Let <math alttext="\\alpha\\in(0,1)" display="inline"><annotation encoding="application/x-tex">\\alpha\\in(0,1)</annotation></math> be given.</p>
<table id="S1.E1" class="ltx_equation ltx_eqn_table">
  <tbody><tr>
    <td><math alttext="x=1" display="block"><annotation encoding="application/x-tex">x=1</annotation></math></td>
    <td class="ltx_eqn_eqno ltx_align_middle">
      <span class="ltx_tag ltx_tag_equation ltx_align_left">(1.1)</span>
    </td>
  </tr></tbody>
</table>
<p>Done.</p>
</body></html>
"""

STRUCTURED_HTML = """
<html><body><article>
<div>arXiv:2606.25586v1</div>
<h1 class="ltx_title ltx_title_document">A Mathematical Paper</h1>
<div class="ltx_authors">
  <span class="ltx_creator ltx_role_author">
    <span class="ltx_personname">Ada Lovelace</span>
    <span class="ltx_contact ltx_role_address">Analytical Engine Institute</span>
    <span class="ltx_contact ltx_role_email">ada@example.org</span>
  </span>
</div>
<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract.</h6><p>Main result.</p></div>
<p><span class="ltx_text ltx_font_bold">Keywords and Phrases:</span> analysis, quadrature.</p>
<p><span class="ltx_text ltx_font_bold">Mathematics Subject Classification 2020:</span> 41A55, 65D30.</p>
</article></body></html>
"""

COMBINED_AUTHORS_HTML = """
<html><body><article>
<div>References include arXiv:1111.22222.</div>
<h1 class="ltx_title ltx_title_document">Combined Authors</h1>
<div class="ltx_authors"><span class="ltx_creator ltx_role_author">
<span class="ltx_personname">
Zirui Deng<sup>1</sup> Jingbo Sun<sup>1</sup> and Fei Wang<sup>1,∗</sup>
<br/><sup>1</sup><span>School of Mathematics<br/><sup>∗</sup>Correspondence: fei@example.org</span>
</span>
</span></div>
</article></body></html>
"""


def test_inline_math_replaced():
    md = _convert_html(MINIMAL_HTML)
    assert "$\\alpha\\in(0,1)$" in md


def test_display_equation_table_replaced():
    md = _convert_html(MINIMAL_HTML)
    assert "$$x=1$$" in md


def test_display_equation_keeps_number_and_source_id():
    md = _convert_html(MINIMAL_HTML)
    assert "[Equation metadata: source_id=S1.E1; number=(1.1)]" in md


def test_equation_group_keeps_number_source_id_and_rows():
    html = r"""
    <html><body>
    <table id="Sx1.EGx1" class="ltx_equationgroup ltx_eqn_eqnarray ltx_eqn_table">
      <tr>
        <td><math alttext="K(t)"></math></td>
        <td><math alttext="=A(t)"></math></td>
        <td><span class="ltx_tag ltx_tag_equation">(1.5)</span></td>
      </tr>
      <tr><td></td><td><math alttext="=B(t)"></math></td><td></td></tr>
    </table>
    </body></html>
    """
    md = _convert_html(html)
    assert "[Equation metadata: source_id=Sx1.EGx1; number=(1.5)]" in md
    assert "$K(t)$" in md
    assert "$=A(t)$" in md
    assert "$=B(t)$" in md


def test_no_cid_artifacts():
    md = _convert_html(MINIMAL_HTML)
    assert "(cid:" not in md


def test_structured_document_metadata_and_headings():
    md = _convert_html(STRUCTURED_HTML)
    assert md.startswith("---\n")
    assert 'arxiv_id: "2606.25586v1"' in md
    assert 'title: "A Mathematical Paper"' in md
    assert '  - name: "Ada Lovelace"' in md
    assert '    affiliation: "Analytical Engine Institute"' in md
    assert '    email: "ada@example.org"' in md
    assert '  - "analysis"' in md
    assert '  - "quadrature"' in md
    assert '  - "41A55"' in md
    assert "# A Mathematical Paper" in md
    assert "## Authors" in md
    assert "## Abstract" in md
    assert "## Keywords" in md
    assert "## Mathematics Subject Classification 2020" in md


def test_missing_document_metadata_uses_explicit_empty_values():
    md = _convert_html(MINIMAL_HTML, arxiv_id="2606.28184")
    assert 'arxiv_id: "2606.28184"' in md
    assert 'title: ""' in md
    assert "authors: []" in md
    assert "keywords: []" in md
    assert "msc_2020: []" in md


def test_requested_arxiv_id_wins_over_ids_mentioned_in_document():
    md = _convert_html(COMBINED_AUTHORS_HTML, arxiv_id="2606.29440")
    assert 'arxiv_id: "2606.29440"' in md
    assert 'arxiv_id: "1111.22222"' not in md


def test_combined_author_layout_is_split_into_people():
    md = _convert_html(COMBINED_AUTHORS_HTML, arxiv_id="2606.29440")
    assert '  - name: "Zirui Deng"' in md
    assert '  - name: "Jingbo Sun"' in md
    assert '  - name: "Fei Wang"' in md
    assert md.count('affiliation: "School of Mathematics"') == 3
    assert '    email: "fei@example.org"' in md


def test_fetch_and_convert_calls_arxiv_html5():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.text = MINIMAL_HTML
    with patch("topic_overviews.arxiv_to_md.requests.get", return_value=mock_response) as mock_get:
        result = fetch_and_convert("2606.28184")
    url = mock_get.call_args[0][0]
    assert "arxiv.org/html/2606.28184" in url
    assert "$\\alpha\\in(0,1)$" in result


def test_fetch_retries_twice_then_raises():
    import requests
    with patch("topic_overviews.arxiv_to_md.requests.get", side_effect=requests.RequestException("timeout")) as mock_get, \
         patch("topic_overviews.arxiv_to_md.time.sleep") as mock_sleep:
        try:
            fetch_and_convert("0000.00000")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "0000.00000" in str(e)
    assert mock_get.call_count == 3  # 1 initial + 2 retries
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(3)


def test_fetch_succeeds_on_second_attempt():
    import requests
    mock_ok = MagicMock()
    mock_ok.raise_for_status = MagicMock()
    mock_ok.text = MINIMAL_HTML
    with patch("topic_overviews.arxiv_to_md.requests.get", side_effect=[requests.RequestException("timeout"), mock_ok]), \
         patch("topic_overviews.arxiv_to_md.time.sleep"):
        result = fetch_and_convert("2606.28184")
    assert "$\\alpha\\in(0,1)$" in result
