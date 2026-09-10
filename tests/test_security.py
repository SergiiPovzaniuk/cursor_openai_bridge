import pytest
from fastapi import HTTPException

from host import security


class FakeConfig:
    dev_mode = False
    bearer_token = "secret123"
    allowlist_ips = ()
    rate_limit_per_minute = 2
    relay_token = "reltoken"


def test_require_bearer_ok(monkeypatch):
    monkeypatch.setattr(security, "CONFIG", FakeConfig())
    security.require_bearer("Bearer secret123")


def test_require_bearer_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(security, "CONFIG", FakeConfig())
    with pytest.raises(HTTPException):
        security.require_bearer("Bearer wrong")


def test_require_bearer_rejects_missing(monkeypatch):
    monkeypatch.setattr(security, "CONFIG", FakeConfig())
    with pytest.raises(HTTPException):
        security.require_bearer(None)


def test_require_bearer_rejects_unconfigured_token(monkeypatch):
    class NoAuthConfig(FakeConfig):
        bearer_token = ""

    monkeypatch.setattr(security, "CONFIG", NoAuthConfig())
    with pytest.raises(HTTPException):
        security.require_bearer(None)


def test_dev_mode_allows_empty_tokens(monkeypatch):
    class DevConfig(FakeConfig):
        bearer_token = ""
        relay_token = ""
        dev_mode = True

    monkeypatch.setattr(security, "CONFIG", DevConfig())
    security.require_bearer(None)
    assert security.check_relay_token(None)


def test_check_relay_token(monkeypatch):
    monkeypatch.setattr(security, "CONFIG", FakeConfig())
    assert security.check_relay_token("reltoken")
    assert not security.check_relay_token("wrong")
    assert not security.check_relay_token(None)


class FakeClient:
    host = "1.2.3.4"


class FakeRequest:
    client = FakeClient()


def test_rate_limit_blocks_after_threshold(monkeypatch):
    monkeypatch.setattr(security, "CONFIG", FakeConfig())
    security._buckets.clear()
    req = FakeRequest()
    security.check_rate_limit(req)
    security.check_rate_limit(req)
    with pytest.raises(HTTPException):
        security.check_rate_limit(req)
