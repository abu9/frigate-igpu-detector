import builtins

from frigate_igpu_detector import cli
from frigate_igpu_detector.cli import _parse_env_blob, _parse_proc_environ


def test_parse_env_blob():
    blob = 'A=1 B=two C="x y"'
    assert _parse_env_blob(blob) == {"A": "1", "B": "two", "C": "x y"}


def test_parse_env_blob_ignores_tokens_without_equals():
    assert _parse_env_blob("A=1 junk B=2") == {"A": "1", "B": "2"}


def test_parse_proc_environ():
    raw = b"A=1\0B=two\0C=x=y\0"
    assert _parse_proc_environ(raw) == {"A": "1", "B": "two", "C": "x=y"}


def test_no_subcommand_prints_help(capsys):
    assert cli.main([]) == 0
    output = capsys.readouterr().out
    assert "usage: fzd" in output
    assert "fzd serve" in output


def test_preflight_fails_without_any_backend(monkeypatch, tmp_path, capsys):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("ncnn", "openvino"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(cli, "_deployed_environment", lambda unit: ({}, "test"))
    monkeypatch.setenv("FRIGATE_ZMQ_BACKEND", "ncnn-vulkan")
    monkeypatch.setenv("FRIGATE_ZMQ_BIND", "tcp://127.0.0.1:5555")
    monkeypatch.setenv("FRIGATE_ZMQ_MODEL_DIR", str(tmp_path))

    assert cli._preflight() == 1
    output = capsys.readouterr().out
    assert "no usable" not in output
    assert "preflight failed" in output
