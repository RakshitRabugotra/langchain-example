import chardet
# Typing
from os import PathLike


def get_file_encoding(file_path: PathLike):
    """Detect encoding of `file_path`"""
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        detected_encoding = result["encoding"]
        return detected_encoding
