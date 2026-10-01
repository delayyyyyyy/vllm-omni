# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova mixed-traffic warmup configuration and request coverage."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.lora.manager import LoRABackend
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    SenseNovaU1Pipeline,
    _parse_mixed_warmup_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_mixed_warmup_covers_selected_task_and_resolution_shapes(monkeypatch):
    host = object.__new__(SenseNovaU1Pipeline)
    host._mixed_warmup = _parse_mixed_warmup_config(
        {"resolutions": [[1024, 1024], [1536, 1536]], "text_to_text": True, "image_to_text": True},
        grid_factor=16,
    )
    host._mixed_warmup_done = False
    calls = []

    def parse(request):
        sampling = request.sampling_params
        return SimpleNamespace(
            prompt=request.prompts[0]["prompt"],
            image_size=(sampling.width, sampling.height),
            num_steps=sampling.num_inference_steps,
            extra_args=sampling.extra_args,
        )

    monkeypatch.setattr(host, "_parse_request", parse)
    monkeypatch.setattr(
        host,
        "_forward_text",
        lambda params, images: calls.append(("text", bool(images), params.extra_args["max_tokens"])),
    )
    monkeypatch.setattr(
        host,
        "_forward_t2i",
        lambda params: calls.append(
            ("image", params.image_size, params.num_steps, params.extra_args["think"], params.extra_args["cfg_scale"])
        ),
    )

    SenseNovaU1Pipeline._warm_mixed_shapes(host)
    SenseNovaU1Pipeline._warm_mixed_shapes(host)

    assert calls == [
        ("text", False, 1),
        ("text", True, 1),
        ("image", (1024, 1024), 1, False, 4.0),
        ("image", (1536, 1536), 1, False, 4.0),
    ]


@pytest.mark.parametrize(
    "profile",
    [
        {"resolutions": [[1025, 1024]]},
        {"resolutions": [[1024, 1024], [1024, 1024]]},
        {"resolutions": [[1024, 1024]] * 4},
        {"resolutions": [[True, 1024]]},
        {"resolutions": [[4096, 4112]]},
        {"resolutions": [], "text_to_text": 1},
        {"resolutions": [[1024, 1024]], "cfg_scale": -1},
        {"resolutions": [[1024, 1024]], "cfg_scale": True},
        {"unknown": True},
        {},
    ],
)
def test_mixed_warmup_rejects_invalid_or_unbounded_profiles(profile):
    with pytest.raises(ValueError):
        _parse_mixed_warmup_config(profile, grid_factor=16)


def test_mixed_warmup_uses_distilled_lora_cfg(monkeypatch):
    host = object.__new__(SenseNovaU1Pipeline)
    host._mixed_warmup = _parse_mixed_warmup_config({"resolutions": [[1024, 1024]]}, grid_factor=16)
    host._mixed_warmup_done = False
    host.od_config = SimpleNamespace(lora_backend=LoRABackend.DISTILL, lora_path="adapter")
    seen = []
    monkeypatch.setattr(host, "_parse_request", lambda request: request.sampling_params)
    monkeypatch.setattr(host, "_forward_t2i", lambda params: seen.append(params.extra_args["cfg_scale"]))

    SenseNovaU1Pipeline._warm_mixed_shapes(host)

    assert seen == [1.0]
