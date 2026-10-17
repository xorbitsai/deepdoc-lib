"""Keep Zhipu vision working without imposing the legacy SDK's PyJWT cap."""

import base64
import importlib.util
import json
import re
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from packaging.requirements import Requirement


@pytest.fixture
def vision_module():
    # Load only this adapter: dependency/signature tests need no OCR models,
    # parser initialization or network model downloads.
    path = Path(__file__).parents[1] / "deepdoc/depend/simple_cv_model.py"
    spec = importlib.util.spec_from_file_location("vision_sdk_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("image", [b"image-bytes", BytesIO(b"image-bytes")])
def test_zhipu_vision_request_preserves_endpoint_and_payload(vision_module, monkeypatch, image):
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "vision-result",
                "created": 1,
                "model": "glm-4v",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": " described "}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client_type = vision_module.ZhipuAiClient
    monkeypatch.setenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        monkeypatch.setattr(vision_module, "ZhipuAiClient", lambda **kwargs: client_type(http_client=transport, **kwargs))
        model = vision_module.Zhipu4V("test-id.test-secret-with-at-least-32-bytes")
        assert model.describe_with_prompt(image, "Describe this image") == "described"
    (request,) = seen
    assert str(request.url) == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert request.headers["Authorization"]
    payload = json.loads(request.content)
    assert payload["model"] == "glm-4v"
    assert payload["max_tokens"] == 1000
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "Describe this image"},
        # The official SDK converts a data URI to the provider's base64 field.
        {"type": "image_url", "image_url": {"url": base64.b64encode(b"image-bytes").decode()}},
    ]


def test_dependency_metadata_does_not_require_legacy_zhipuai():
    project = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert '"zhipuai' not in project
    (sdk,) = [Requirement(value) for value in re.findall(r'"(zai-sdk[^"\n]*)"', project)]
    assert "0.0.3.4" not in sdk.specifier
    assert "0.0.4" in sdk.specifier
