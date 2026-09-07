from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse


class OpenAIError(Exception):
    def __init__(self, message: str, *, status: int = 400, type_: str = "invalid_request_error", code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.type_ = type_
        self.code = code

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type_, "param": None, "code": self.code}}

    def response(self) -> JSONResponse:
        return JSONResponse(status_code=self.status, content=self.body())


def context_length_exceeded(message: str) -> OpenAIError:
    return OpenAIError(message, status=400, type_="invalid_request_error", code="context_length_exceeded")


def not_implemented(message: str) -> OpenAIError:
    return OpenAIError(message, status=501, type_="invalid_request_error", code="not_implemented")


def unauthorized(message: str = "Invalid API key") -> HTTPException:
    return HTTPException(status_code=401, detail={"error": {"message": message, "type": "invalid_request_error", "code": "invalid_api_key"}})


def rate_limited(message: str = "Rate limit exceeded") -> HTTPException:
    return HTTPException(status_code=429, detail={"error": {"message": message, "type": "rate_limit_error", "code": "rate_limit_exceeded"}})


def conversation_busy(message: str = "Conversation already has an active turn") -> OpenAIError:
    return OpenAIError(message, status=409, type_="conversation_busy", code="agent_busy")
