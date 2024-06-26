"""Test `serenityff.charge.utils.exceptions`."""

import pytest

from serenityff.charge.utils.exceptions import (
    DataDownloadError,
    DataExtractionError,
    DataIncompleteError,
    ExtractionError,
    NotInitializedError,
)


@pytest.mark.parametrize(
    "exc",
    [
        ExtractionError,
        NotInitializedError,
        DataDownloadError,
        DataExtractionError,
        DataIncompleteError,
    ],
)
def test_extraction_error(exc: Exception) -> None:
    """Test `ExtractionError` and `NotInitializedError`.

    Args:
        exc (Exception): Exception to test.
    """
    msg = "Exception Message"

    with pytest.raises(exc) as excinfo:
        raise exc(msg)

    assert str(excinfo.value) == msg
