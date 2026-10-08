"""Request retries must recover without replaying a delivered tool stream."""

import json
from email.utils import formatdate

import pytest
import requests

from langslice.providers import openai_oauth as oauth


class Response:
    def __init__(self, status=200, text="temporary failure", headers=None, broken=False):
        self.status_code = status
        self.text = text
        self.headers = headers or {}
        self.closed = False
        self.broken = broken

    def close(self):
        self.closed = True

    def iter_lines(self):
        yield b'data: {"type":"response.output_item.done","item":{"type":"function_call"}}'
        if self.broken:
            raise requests.ConnectionError("stream interrupted")
        yield b"data: [DONE]"


@pytest.fixture
def transport(monkeypatch):
    calls, waits = [], []
    outcomes = []
    monkeypatch.setattr(oauth, "load_credentials", lambda: oauth.Creds("secret", "refresh-secret"))
    monkeypatch.setattr(oauth.random, "uniform", lambda a, b: 0)
    monkeypatch.setattr(oauth.time, "sleep", waits.append)

    def post(url, **kwargs):
        calls.append((url, kwargs))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(oauth.requests, "post", post)
    return outcomes, calls, waits


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 507])
def test_recovers_with_same_request_and_closes_responses(transport, caplog, status):
    outcomes, calls, waits = transport
    failed, success = Response(status), Response(headers={"x-codex-primary-used-percent": "1"})
    outcomes.extend([failed, success])
    events = list(oauth.stream_events({"input": ["original"]}, session_id="stable"))
    assert len(calls) == 2 and calls[0] == calls[1]
    assert waits == [2]
    assert failed.closed and success.closed
    assert events[0] == {"type": "langslice.quota", "quota": {"primary_used_percent": "1"}}
    assert len(events) == 2
    assert "retry 1/3" in caplog.text
    assert "secret" not in caplog.text


def test_stops_after_three_retries(transport):
    outcomes, calls, waits = transport
    failures = [Response(507) for _ in range(4)]
    outcomes.extend(failures)
    with pytest.raises(RuntimeError, match="507"):
        oauth.stream_events({})
    assert len(calls) == 4
    assert waits == [2, 4, 8]
    assert all(response.closed for response in failures)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_permanent_http_errors_are_not_retried(transport, monkeypatch, status):
    outcomes, calls, waits = transport
    monkeypatch.setattr(oauth, "load_credentials", lambda: oauth.Creds("secret"))
    outcomes.append(Response(status))
    with pytest.raises(RuntimeError, match=str(status)):
        oauth.stream_events({})
    assert len(calls) == 1 and not waits


@pytest.mark.parametrize(
    "code", ["insufficient_quota", "usage_limit_reached", "billing_hard_limit_reached"]
)
def test_hard_quota_errors_are_not_retried(transport, code):
    outcomes, calls, waits = transport
    outcomes.append(Response(429, json.dumps({"error": {"code": code}})))
    with pytest.raises(RuntimeError, match=code):
        oauth.stream_events({})
    assert len(calls) == 1 and not waits


@pytest.mark.parametrize("error", [requests.ConnectionError, requests.Timeout])
def test_connection_failure_retries(transport, error):
    outcomes, calls, waits = transport
    outcomes.extend([error("connection failed"), Response()])
    list(oauth.stream_events({}))
    assert len(calls) == 2 and waits == [2]


def test_certificate_error_is_not_retried(transport):
    outcomes, calls, waits = transport
    outcomes.append(requests.exceptions.SSLError("certificate invalid"))
    with pytest.raises(requests.exceptions.SSLError):
        oauth.stream_events({})
    assert len(calls) == 1 and not waits


def test_retry_after_seconds_and_http_date(transport, monkeypatch):
    outcomes, _, waits = transport
    monkeypatch.setattr(oauth.time, "time", lambda: 1000)
    outcomes.extend(
        [
            Response(429, headers={"Retry-After": "10"}),
            Response(503, headers={"Retry-After": formatdate(1020, usegmt=True)}),
            Response(),
        ]
    )
    list(oauth.stream_events({}))
    assert waits == [10, 20]


@pytest.mark.parametrize("header", ["bad", "nan", "-3"])
def test_invalid_retry_after_uses_backoff(transport, header):
    outcomes, _, waits = transport
    outcomes.extend([Response(503, headers={"Retry-After": header}), Response()])
    list(oauth.stream_events({}))
    assert waits == [2]


def test_does_not_retry_before_long_server_delay(transport):
    outcomes, calls, waits = transport
    outcomes.append(Response(429, headers={"Retry-After": "120"}))
    with pytest.raises(RuntimeError, match="429"):
        oauth.stream_events({})
    assert len(calls) == 1 and not waits


def test_refresh_once_without_resetting_retry_budget(transport, monkeypatch):
    outcomes, calls, waits = transport
    refreshed = []

    def refresh(creds):
        refreshed.append(creds)
        return oauth.Creds("new-secret", "new-refresh-secret")

    monkeypatch.setattr(oauth, "refresh", refresh)
    failures = [Response(code) for code in [507, 401, 507, 507, 507]]
    outcomes.extend(failures)
    with pytest.raises(RuntimeError, match="507"):
        oauth.stream_events({})
    assert len(refreshed) == 1 and len(calls) == 5
    assert waits == [2, 4, 8]
    assert all(response.closed for response in failures)
    assert calls[2][1]["headers"]["Authorization"] == "Bearer new-secret"


def test_second_401_stops(transport, monkeypatch):
    outcomes, calls, waits = transport
    monkeypatch.setattr(oauth, "refresh", lambda creds: creds)
    outcomes.extend([Response(401), Response(401)])
    with pytest.raises(RuntimeError, match="401"):
        oauth.stream_events({})
    assert len(calls) == 2 and not waits


def test_never_replays_partially_delivered_stream(transport):
    outcomes, calls, waits = transport
    response = Response(broken=True)
    outcomes.append(response)
    stream = oauth.stream_events({})
    next(stream)  # quota
    assert next(stream)["item"]["type"] == "function_call"
    with pytest.raises(requests.ConnectionError, match="stream interrupted"):
        next(stream)
    assert len(calls) == 1 and not waits
    assert response.closed
