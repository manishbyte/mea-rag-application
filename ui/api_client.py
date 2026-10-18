
"""HTTP client for the MEA RAG FastAPI backend."""

import os
from typing import Any
from uuid import UUID

import httpx


DEFAULT_API_URL = "http://127.0.0.1:8000"


class APIError(Exception):
    """Readable error returned by the backend or network layer."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class APIClient:
    def __init__(self, base_url: str | None = None, timeout: float = 90.0):
        self.base_url = (
            base_url or os.getenv("MEA_API_URL") or DEFAULT_API_URL
        ).rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        token: str | None = None,
        **kwargs: Any,
    ) -> Any:
        headers = dict(kwargs.pop("headers", {}) or {})
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = httpx.request(
                method=method,
                url=f"{self.base_url}{path}",
                headers=headers,
                timeout=self.timeout,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise APIError(
                "The backend took too long to respond. Please try again."
            ) from exc
        except httpx.ConnectError as exc:
            raise APIError(
                "Cannot connect to the MEA API. Check that FastAPI is running "
                "and MEA_API_URL is correct."
            ) from exc
        except httpx.RequestError as exc:
            raise APIError(f"Network error: {exc}") from exc

        if response.status_code >= 400:
            try:
                body = response.json()
                detail = body.get("detail", body)
                if isinstance(detail, list):
                    detail = "; ".join(
                        str(item.get("msg", item)) if isinstance(item, dict)
                        else str(item)
                        for item in detail
                    )
                message = str(detail)
            except ValueError:
                message = response.text or "Unknown API error"

            if response.status_code == 401:
                message = "Your session has expired or your credentials are invalid. Please log in again."
            elif response.status_code == 403:
                message = "You do not have permission to perform this action."
            elif response.status_code == 404:
                message = "The requested resource was not found."
            elif response.status_code == 429:
                message = "Too many requests. Please wait and try again."
            elif response.status_code >= 500:
                message = "The backend encountered an error. Please try again later."

            raise APIError(message, response.status_code)

        if response.status_code == 204 or not response.content:
            return None

        try:
            return response.json()
        except ValueError as exc:
            raise APIError("The API returned an invalid JSON response.") from exc

    def register(self, email: str, password: str) -> dict:
        return self._request(
            "POST",
            "/auth/register",
            json={"email": email, "password": password},
        )

    def login(self, email: str, password: str) -> dict:
        return self._request(
            "POST",
            "/auth/login",
            json={"email": email, "password": password},
        )

    def me(self, token: str) -> dict:
        return self._request("GET", "/auth/me", token=token)

    def logout(self, token: str) -> None:
        self._request("POST", "/auth/logout", token=token)

    def list_conversations(self, token: str) -> list[dict]:
        data = self._request(
            "GET", "/chat/conversations", token=token
        )
        # Support either a raw list or a wrapped response.
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("conversations", data.get("items", []))
        return []

    def get_conversation(self, token: str, conversation_id: UUID | str) -> dict:
        return self._request(
            "GET",
            f"/chat/conversations/{conversation_id}",
            token=token,
        )

    def send_message(
        self,
        token: str,
        question: str,
        conversation_id: UUID | str | None = None,
    ) -> dict:
        payload: dict[str, Any] = {"question": question}
        if conversation_id:
            payload["conversation_id"] = str(conversation_id)

        return self._request(
            "POST", "/chat", token=token, json=payload
        )


def get_client() -> APIClient:
    return APIClient()
