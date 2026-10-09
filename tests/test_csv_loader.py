from io import BytesIO

import pytest

from src.csv_loader import CSVLoadError, load_csv


def upload(contents: bytes, name: str = "data.csv") -> BytesIO:
    file = BytesIO(contents)
    file.name = name
    return file


def test_loads_valid_csv():
    dataframe = load_csv(upload(b"name,score\nAda,10\n"))
    assert dataframe.to_dict("records") == [{"name": "Ada", "score": 10}]


def test_rejects_non_csv_extension():
    with pytest.raises(CSVLoadError, match="Only CSV"):
        load_csv(upload(b"name\nAda\n", "data.txt"))


def test_rejects_empty_file():
    with pytest.raises(CSVLoadError, match="empty"):
        load_csv(upload(b""))


def test_rejects_malformed_csv():
    with pytest.raises(CSVLoadError, match="valid CSV"):
        load_csv(upload(b'name,value\n"unclosed,1'))


def test_rejects_encoding_error():
    with pytest.raises(CSVLoadError, match="encoding"):
        load_csv(upload(b"name\n\xff\xfe\n"))


def test_rejects_file_over_size_limit():
    with pytest.raises(CSVLoadError, match="exceeds"):
        load_csv(upload(b"a\n1\n"), max_size_mb=0.000001)
