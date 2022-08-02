class Error(Exception):
    """
    Bass class for Exceptions in this package.
    """

    pass


class ExtractionError(Error):
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


class NotInitializedError(Error):
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
