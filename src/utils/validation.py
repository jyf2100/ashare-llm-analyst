"""
Validation utilities for stock analysis.
"""

import re
from typing import Any, Dict, List, Optional, Union

from src.core.exceptions import ValidationError


# Stock code patterns
STOCK_CODE_PATTERN = re.compile(r"^[0-9]{6}$")
STOCK_CODE_WITH_PREFIX = re.compile(r"^(sh|sz|SH|SZ)[.]?[0-9]{6}$")


def validate_stock_code(code: str) -> str:
    """
    Validate and normalize stock code.

    Args:
        code: Stock code (e.g., "000001", "sh.000001", "sh000001")

    Returns:
        Normalized stock code (e.g., "sh.000001")

    Raises:
        ValidationError: If code is invalid
    """
    if not code or not isinstance(code, str):
        raise ValidationError(f"Invalid stock code: {code}")

    code = code.strip()

    # Check with prefix patterns
    if STOCK_CODE_WITH_PREFIX.match(code):
        # Normalize to sh./sz. prefix format
        if re.match(r"^(SH|SZ)[.]", code):
            return code.lower().replace("sh", "sh.").replace("sz", "sz.")
        elif code.startswith(("sh", "sz")) and "." not in code:
            return f"{code[:2]}.{code[2:]}"
        return code.lower()

    # Check 6-digit pattern
    if STOCK_CODE_PATTERN.match(code):
        # Add prefix based on first digit
        first_digit = code[0]
        if first_digit == "6":
            return f"sh.{code}"
        elif first_digit in ("0", "2", "3"):
            return f"sz.{code}"
        else:
            raise ValidationError(f"Unknown stock code format: {code}")

    raise ValidationError(f"Invalid stock code format: {code}")


def validate_positive_number(
    value: Any,
    name: str = "value",
    allow_zero: bool = False,
) -> float:
    """
    Validate value is a positive number.

    Args:
        value: Value to validate
        name: Name for error messages
        allow_zero: Whether zero is allowed

    Returns:
        Validated float value

    Raises:
        ValidationError: If validation fails
    """
    try:
        num_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number: {value}")

    min_value = 0 if allow_zero else 1e-10

    if num_value < min_value:
        requirement = "non-negative" if allow_zero else "positive"
        raise ValidationError(f"{name} must be {requirement}: {value}")

    return num_value


def validate_range(
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "value",
) -> float:
    """
    Validate value is within range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value (None = no minimum)
        max_value: Maximum allowed value (None = no maximum)
        name: Name for error messages

    Returns:
        Validated float value

    Raises:
        ValidationError: If validation fails
    """
    try:
        num_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number: {value}")

    if min_value is not None and num_value < min_value:
        raise ValidationError(
            f"{name} must be >= {min_value}, got {num_value}"
        )

    if max_value is not None and num_value > max_value:
        raise ValidationError(
            f"{name} must be <= {max_value}, got {num_value}"
        )

    return num_value


def validate_required_fields(
    data: Dict[str, Any],
    required_fields: List[str],
) -> bool:
    """
    Validate that all required fields are present.

    Args:
        data: Dictionary to validate
        required_fields: List of required field names

    Returns:
        True if all fields present

    Raises:
        ValidationError: If any required field is missing
    """
    missing = [field for field in required_fields if field not in data]

    if missing:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing)}"
        )

    # Check for None/empty values
    empty = [
        field
        for field in required_fields
        if data[field] is None
        or (isinstance(data[field], (str, list, dict)) and not data[field])
    ]

    if empty:
        raise ValidationError(
            f"Empty required fields: {', '.join(empty)}"
        )

    return True


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if valid format

    Raises:
        ValidationError: If format is invalid
    """
    pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    if not pattern.match(email):
        raise ValidationError(f"Invalid email format: {email}")

    return True


def validate_percentage(
    value: Any,
    name: str = "value",
    allow_zero: bool = True,
    allow_negative: bool = False,
) -> float:
    """
    Validate value is a percentage (0-100).

    Args:
        value: Value to validate
        name: Name for error messages
        allow_zero: Whether zero is allowed
        allow_negative: Whether negative values are allowed

    Returns:
        Validated float value

    Raises:
        ValidationError: If validation fails
    """
    try:
        num_value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number: {value}")

    if not allow_negative and num_value < 0:
        raise ValidationError(f"{name} cannot be negative: {value}")

    if not allow_zero and num_value == 0:
        raise ValidationError(f"{name} cannot be zero")

    if num_value > 100:
        raise ValidationError(f"{name} cannot exceed 100%: {value}")

    return num_value


def validate_choice(
    value: Any,
    choices: List[Any],
    name: str = "value",
) -> Any:
    """
    Validate value is one of the allowed choices.

    Args:
        value: Value to validate
        choices: List of allowed values
        name: Name for error messages

    Returns:
        Validated value

    Raises:
        ValidationError: If value is not in choices
    """
    if value not in choices:
        raise ValidationError(
            f"{name} must be one of {choices}, got: {value}"
        )

    return value


def validate_length(
    value: Any,
    min_length: int = 0,
    max_length: Optional[int] = None,
    name: str = "value",
) -> str:
    """
    Validate string length.

    Args:
        value: Value to validate (will be converted to string)
        min_length: Minimum length
        max_length: Maximum length (None = no maximum)
        name: Name for error messages

    Returns:
        Validated string

    Raises:
        ValidationError: If validation fails
    """
    str_value = str(value)

    if len(str_value) < min_length:
        raise ValidationError(
            f"{name} must be at least {min_length} characters"
        )

    if max_length is not None and len(str_value) > max_length:
        raise ValidationError(
            f"{name} must be at most {max_length} characters"
        )

    return str_value
