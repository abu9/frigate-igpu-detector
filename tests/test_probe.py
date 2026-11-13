import argparse
import json

import numpy as np
import pytest

from frigate_igpu_detector.probe import decode_reply, parse_crop


def test_decode_reply_multipart():
    array = np.zeros((20, 6), dtype=np.float32)
    parts = [
        json.dumps({"shape": [20, 6], "dtype": "float32"}).encode(),
        array.tobytes(),
    ]
    decoded = decode_reply(parts)
    assert decoded.shape == (20, 6)
    assert decoded.dtype == np.float32


def test_decode_reply_single_frame():
    array = np.ones((20, 6), dtype=np.float32)
    decoded = decode_reply([array.tobytes()])
    assert decoded.shape == (20, 6)
    assert decoded.dtype == np.float32


def test_parse_crop():
    assert parse_crop("0,60,260,330") == (0, 60, 260, 330)
    assert parse_crop(None) is None


def test_parse_crop_rejects_bad_values():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_crop("1,2,3")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_crop("1,2,x,4")
