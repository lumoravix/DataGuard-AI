"""Safe CSV upload and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

DEFAULT_MAX_FILE_SIZE_MB = 20


class CSVLoadError(ValueError):
    """Raised when an uploaded file cannot safely be loaded as CSV data."""


def _file_size_bytes(file: BinaryIO) -> int:
    current_position = file.tell()
    file.seek(0, 2)
    size = file.tell()
    file.seek(current_position)
    return size


def load_csv(file: BinaryIO, max_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB) -> pd.DataFrame:
    """Load a CSV upload after validating its extension and size.

    Uploaded content is treated strictly as data and is never executed.
    """
    if file is None:
        raise CSVLoadError("Please select a CSV file.")
    if max_size_mb <= 0:
        raise ValueError("max_size_mb must be greater than zero.")

    if Path(getattr(file, "name", "")).suffix.lower() != ".csv":
        raise CSVLoadError("Only CSV files (.csv) are accepted.")

    try:
        size = _file_size_bytes(file)
    except (AttributeError, OSError) as error:
        raise CSVLoadError("The uploaded file could not be read.") from error
    if size == 0:
        raise CSVLoadError("The uploaded CSV file is empty.")
    if size > max_size_mb * 1024 * 1024:
        raise CSVLoadError(f"The file exceeds the {max_size_mb} MB upload limit.")

    try:
        file.seek(0)
        return pd.read_csv(file)
    except UnicodeDecodeError as error:
        raise CSVLoadError("The file encoding is unsupported. Please use UTF-8.") from error
    except pd.errors.EmptyDataError as error:
        raise CSVLoadError("The uploaded CSV file contains no data.") from error
    except pd.errors.ParserError as error:
        raise CSVLoadError("The uploaded file is not a valid CSV.") from error
    except (OSError, ValueError) as error:
        raise CSVLoadError("The uploaded file could not be parsed as CSV data.") from error
    finally:
        try:
            file.seek(0)
        except (AttributeError, OSError):
            pass
