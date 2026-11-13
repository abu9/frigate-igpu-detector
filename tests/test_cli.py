from frigate_igpu_detector.cli import _parse_env_blob, _parse_proc_environ


def test_parse_env_blob():
    blob = 'A=1 B=two C="x y"'
    assert _parse_env_blob(blob) == {"A": "1", "B": "two", "C": "x y"}


def test_parse_env_blob_ignores_tokens_without_equals():
    assert _parse_env_blob("A=1 junk B=2") == {"A": "1", "B": "2"}


def test_parse_proc_environ():
    raw = b"A=1\0B=two\0C=x=y\0"
    assert _parse_proc_environ(raw) == {"A": "1", "B": "two", "C": "x=y"}
