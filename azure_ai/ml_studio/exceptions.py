"""
Lawrence McDaniel.

https://lawrencemcdaniel.com

Recognize the errors that the Azure ML SDK v2 raises.
"""

from azure.ai.ml.exceptions import ValidationErrorType, ValidationException
from azure.core.exceptions import ResourceNotFoundError


def is_not_found(error: BaseException | None) -> bool:
    """
    Tell whether an Azure ML lookup failed because the asset doesn't exist.

    The SDK reports a missing asset in more than one way. A lookup by version
    raises ResourceNotFoundError. A lookup by label="latest" raises a
    ValidationException, which some operations (data.get) wrap in an
    MlException. So this checks the error and every error it was raised from.

    Args:
        error: The exception that a get() call raised

    Returns:
        bool: True if the asset doesn't exist
    """
    while error is not None:
        if isinstance(error, ResourceNotFoundError):
            return True
        if isinstance(error, ValidationException) and error.error_type == ValidationErrorType.RESOURCE_NOT_FOUND:
            return True
        error = error.__context__
    return False
