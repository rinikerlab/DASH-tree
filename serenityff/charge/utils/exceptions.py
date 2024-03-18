class ExtractionError(Exception):
    """
    Error thrown if Feature extraction did not work properly.
    Called by Extractor.check_final_csv()
    """

    def __init__(self, message: str) -> None:
        """
        Args:
            message (str): Error Message to be shown.
        """
        super().__init__(message)


class NotInitializedError(Exception):
    """
    Error thrown if Feature extraction did not work properly.
    Called by Extractor.check_final_csv()
    """

    def __init__(self, message: str) -> None:
        """
        Args:
            message (str): Error Message to be shown.
        """
        super().__init__(message)
