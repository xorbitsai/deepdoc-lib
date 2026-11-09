"""Keep Gemini vision working on google-genai without the legacy SDK's protobuf<6 cap."""

import base64
import importlib.util
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def vision_module():
    # Load only this adapter: SDK contract tests need no OCR models,
    # parser initialization or network model downloads.
    path = Path(__file__).parents[1] / "deepdoc/depend/simple_cv_model.py"
    spec = importlib.util.spec_from_file_location("gemini_sdk_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_genai(monkeypatch, genai_module):
    """Make ``from google import genai`` resolve to ``genai_module`` (None = missing)."""
    google = sys.modules.get("google")
    if google is None:
        try:
            import google
        except ImportError:
            google = types.ModuleType("google")
            google.__path__ = []
            monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    if genai_module is None:
        monkeypatch.delattr(google, "genai", raising=False)
    else:
        monkeypatch.setattr(google, "genai", genai_module, raising=False)


class FakeModels:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response
        self.error = error

    def generate_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def fake_genai(monkeypatch):
    state = types.SimpleNamespace(clients=[], models=FakeModels())

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.models = state.models
            state.clients.append(self)

    _install_genai(monkeypatch, types.SimpleNamespace(Client=FakeClient))
    return state


def _png_bytes():
    buf = BytesIO()
    Image.new("RGB", (2, 3), "red").save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("as_stream", [False, True])
def test_gemini_describe_sends_prompt_and_image(vision_module, fake_genai, as_stream):
    fake_genai.models.response = types.SimpleNamespace(text="  a red square \n", prompt_feedback=None)
    data = _png_bytes()
    image = BytesIO(data) if as_stream else data

    model = vision_module.GeminiCV("test-key", "gemini-2.5-flash")

    assert model.describe_with_prompt(image, "Describe this image") == "a red square"
    (client,) = fake_genai.clients
    assert client.kwargs == {"api_key": "test-key"}
    (call,) = fake_genai.models.calls
    assert call["model"] == "gemini-2.5-flash"
    prompt, pil_image = call["contents"]
    assert prompt == "Describe this image"
    assert isinstance(pil_image, Image.Image)
    assert pil_image.size == (2, 3)


def test_gemini_uses_default_prompt(vision_module, fake_genai):
    fake_genai.models.response = types.SimpleNamespace(text="ok", prompt_feedback=None)
    model = vision_module.GeminiCV("test-key", lang="English")

    assert model.describe_with_prompt(_png_bytes()) == "ok"
    (call,) = fake_genai.models.calls
    assert call["model"] == "gemini-pro-vision"
    assert call["contents"][0] == "Please describe this image in detail."


@pytest.mark.parametrize(
    "candidates, prompt_feedback, expected",
    [
        # Blocked prompt: no candidates, reason on prompt_feedback.
        ([], "block_reason=SAFETY", ["finish_reason=None", "prompt_feedback=block_reason=SAFETY"]),
        (None, "block_reason=SAFETY", ["finish_reason=None", "prompt_feedback=block_reason=SAFETY"]),
        # Blocked candidate: reason only on the candidate's finish_reason.
        ([types.SimpleNamespace(finish_reason=types.SimpleNamespace(name="SAFETY"))], None, ["finish_reason=SAFETY", "prompt_feedback=None"]),
    ],
)
def test_gemini_empty_response_keeps_error_contract(vision_module, fake_genai, candidates, prompt_feedback, expected):
    fake_genai.models.response = types.SimpleNamespace(text=None, candidates=candidates, prompt_feedback=prompt_feedback)
    model = vision_module.GeminiCV("test-key")

    result = model.describe_with_prompt(_png_bytes(), "Describe")

    assert result.startswith("**ERROR**: Gemini returned no text")
    for fragment in expected:
        assert fragment in result


@pytest.mark.parametrize("image, expected", [(_png_bytes(), "boom"), ("not-bytes", "image must be BytesIO or bytes")])
def test_gemini_errors_are_returned_not_raised(vision_module, fake_genai, image, expected):
    fake_genai.models.error = RuntimeError("boom")
    model = vision_module.GeminiCV("test-key")

    assert model.describe_with_prompt(image, "Describe") == "**ERROR**: " + expected


def test_gemini_factory_builds_gemini_model(vision_module, fake_genai):
    model = vision_module.create_vision_model({"provider": "gemini", "api_key": "test-key"})

    assert isinstance(model, vision_module.GeminiCV)
    assert model.model_name == "gemini-pro-vision"
    assert fake_genai.clients[0].kwargs == {"api_key": "test-key"}


def test_gemini_missing_sdk_names_the_extra(vision_module, monkeypatch):
    _install_genai(monkeypatch, None)

    with pytest.raises(ImportError, match=r"deepdoc-lib\[gemini\]"):
        vision_module.GeminiCV("test-key")


@pytest.fixture
def gemini_server(monkeypatch):
    """Point the installed SDK at a loopback server (no network) and record requests."""
    genai = pytest.importorskip("google.genai")
    state = types.SimpleNamespace(seen=[], response={})

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state.seen.append((self.path, dict(self.headers), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            body = json.dumps(state.response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1")
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    real_client = genai.Client
    base_url = f"http://127.0.0.1:{server.server_port}/"
    monkeypatch.setattr(genai, "Client", lambda **kwargs: real_client(http_options={"base_url": base_url}, **kwargs))
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


def test_real_google_genai_request_carries_prompt_and_image(vision_module, gemini_server):
    gemini_server.response = {"candidates": [{"content": {"role": "model", "parts": [{"text": " described "}]}, "finishReason": "STOP"}]}
    model = vision_module.GeminiCV("test-key", "gemini-2.5-flash")

    assert model.describe_with_prompt(_png_bytes(), "Describe this image") == "described"

    ((path, headers, payload),) = gemini_server.seen
    assert path.endswith("/models/gemini-2.5-flash:generateContent")
    assert {k.lower(): v for k, v in headers.items()}["x-goog-api-key"] == "test-key"
    # Older SDKs send one user turn per list item, newer ones merge them.
    assert {content.get("role") for content in payload["contents"]} == {"user"}
    text_part, image_part = [part for content in payload["contents"] for part in content["parts"]]
    assert text_part == {"text": "Describe this image"}
    image = image_part.get("inlineData") or image_part["inline_data"]
    assert (image.get("mimeType") or image["mime_type"]).startswith("image/")
    assert Image.open(BytesIO(base64.b64decode(image["data"]))).size == (2, 3)


@pytest.mark.parametrize(
    "response, expected",
    [
        ({"candidates": [{"finishReason": "SAFETY"}]}, "finish_reason=SAFETY"),
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "SAFETY"),
        ({"candidates": []}, "finish_reason=None"),
    ],
)
def test_real_google_genai_blocked_response_reports_reason(vision_module, gemini_server, response, expected):
    gemini_server.response = response
    model = vision_module.GeminiCV("test-key", "gemini-2.5-flash")

    result = model.describe_with_prompt(_png_bytes(), "Describe this image")

    assert result.startswith("**ERROR**: Gemini returned no text")
    assert expected in result
