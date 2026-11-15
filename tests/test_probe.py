from predictor.probe import _endpoints, summarize_body


def test_summarize_body_uses_html_title():
    html = "<html><head><title>Just a\n  moment...</title></head><body>x</body></html>"
    assert summarize_body(html) == "HTML <title>Just a moment...</title>"


def test_summarize_body_truncates_text():
    assert summarize_body('{"detail": "nope"}') == '{"detail": "nope"}'
    assert summarize_body("a" * 400, limit=10) == "a" * 10 + "…"


def test_endpoints_cover_info_meta_and_first_page():
    urls = [url for _, url in _endpoints("biweekly-contest-193")]
    assert urls[0].endswith("/contest/api/info/biweekly-contest-193/")
    assert urls[1].endswith("/contest/api/ranking/biweekly-contest-193/")
    assert urls[2].endswith("/contest/api/ranking/biweekly-contest-193/?pagination=1&region=global")
