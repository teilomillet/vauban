# SPDX-FileCopyrightText: 2026 Teilo Millet
# SPDX-License-Identifier: Apache-2.0

"""Network-facing validation helpers."""

from urllib.parse import SplitResult, urlsplit

_ALLOWED_HTTP_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def validate_http_url(url: str, *, context: str) -> str:
    """Require an absolute HTTP(S) URL for user-controlled network inputs.

    Args:
        url: URL to validate.
        context: Human-readable label used in error messages.

    Returns:
        The original URL when validation succeeds.

    Raises:
        ValueError: If the URL is not absolute HTTP(S).
    """
    parsed: SplitResult = urlsplit(url)
    if parsed.scheme not in _ALLOWED_HTTP_SCHEMES:
        schemes = " or ".join(sorted(_ALLOWED_HTTP_SCHEMES))
        msg = f"{context} must use {schemes}, got {url!r}"
        raise ValueError(msg)
    if not parsed.netloc:
        msg = f"{context} must be an absolute URL, got {url!r}"
        raise ValueError(msg)
    return url
