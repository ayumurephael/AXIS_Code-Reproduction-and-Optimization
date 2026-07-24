from __future__ import annotations

import urllib.error

import pytest

from .geval_resilient import (
    PermanentAPIError,
    RetryExhaustedError,
    run_retrying,
)


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example.invalid/chat/completions",
        code=code,
        msg="test error",
        hdrs=None,
        fp=None,
    )


def test_permanent_http_error_fails_without_retrying():
    attempts = 0

    def worker(_item):
        nonlocal attempts
        attempts += 1
        raise http_error(401)

    with pytest.raises(PermanentAPIError, match="permanent API error 401"):
        list(run_retrying(
            ["row"],
            worker,
            workers=1,
            phase="primary",
            max_rounds=3,
            sleep_fn=lambda _seconds: None,
        ))
    assert attempts == 1


def test_transient_http_error_retries_then_succeeds():
    attempts = 0

    def worker(item):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http_error(429)
        return item

    results = list(run_retrying(
        ["row"],
        worker,
        workers=1,
        phase="primary",
        max_rounds=3,
        sleep_fn=lambda _seconds: None,
    ))
    assert results == ["row"]
    assert attempts == 2


def test_transient_http_error_has_bounded_retry_rounds():
    attempts = 0

    def worker(_item):
        nonlocal attempts
        attempts += 1
        raise http_error(503)

    with pytest.raises(RetryExhaustedError, match="exhausted 2 retry rounds"):
        list(run_retrying(
            ["row"],
            worker,
            workers=1,
            phase="fallback",
            max_rounds=2,
            sleep_fn=lambda _seconds: None,
        ))
    assert attempts == 2
