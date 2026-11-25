from fastapi import WebSocket
import math
import re


LOCAL_BROWSER_ORIGINS = (
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
    "http://localhost:8000", "http://127.0.0.1:8000",
)
_SENSITIVE_FIELD = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    r"secret|password|passwd|private[_-]?key|authorization|credential|cookie)"
)


def redact_text(value: str, secrets=()) -> str:
    """Remove known credentials and common credential forms from UI diagnostics."""
    if "Traceback (most recent call last)" in value:
        return "Operation failed; details are available in local server logs."
    known = sorted({item for item in secrets if isinstance(item, str) and item}, key=len, reverse=True)
    if known:
        value = re.sub("|".join(re.escape(item) for item in known), "[REDACTED]", value)
    value = re.sub(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                   "[REDACTED]", value, flags=re.DOTALL)
    value = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", "[REDACTED]", value)
    value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|AIza[A-Za-z0-9_-]{35})",
                   "[REDACTED]", value)
    value = re.sub(
        r'''(?i)(["']?(?:api[_-]?key|token|secret|password|passwd|authorization)["']?\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;}\]]+)''',
        r"\1[REDACTED]", value,
    )
    value = re.sub(r"(?i)(https?://)[^\s/:@]+:[^\s/@]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)([?&](?:api[_-]?key|access_token|token|secret|signature|sig)=)[^&\s]+",
                   r"\1[REDACTED]", value)
    return value


def public_payload(value, secrets=()):
    """Sanitize HTTP, live events and replay snapshots without mutating state."""
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if _SENSITIVE_FIELD.search(str(key)) else public_payload(item, secrets))
            for key, item in value.items()
            if str(key).lower() not in {"traceback", "stack_trace", "exc_info"}
        }
    if isinstance(value, (list, tuple)):
        return [public_payload(item, secrets) for item in value]
    if isinstance(value, BaseException):
        return "Operation failed; details are available in local server logs."
    return redact_text(value, secrets) if isinstance(value, str) else value


def websocket_origin_allowed(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    if origin is not None:
        return origin in LOCAL_BROWSER_ORIGINS
    # Non-browser clients do not send Origin. Retain local CLI and TestClient.
    client = websocket.client
    return client is not None and client.host in {"127.0.0.1", "::1", "localhost", "testclient"}


def json_safe(value):
    """Represent unavailable numeric diagnostics as JSON null without mutation."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(redact_text(message))

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(json_safe(public_payload(message)))
            except:
                # Connection dropped
                pass

manager = ConnectionManager()
