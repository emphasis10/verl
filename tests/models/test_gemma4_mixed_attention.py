# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from verl.models.transformers import gemma4_mixed_attention as mixed


class DummyGemma4Attention(torch.nn.Module):
    def __init__(self, *, is_sliding: bool = False):
        super().__init__()
        self.config = SimpleNamespace(model_type="gemma4_text")
        self.is_sliding = is_sliding
        self.layer_idx = 0


def test_registers_gemma4_mixed_attention_backend():
    assert mixed.ATTN_IMPLEMENTATION_NAME in ALL_ATTENTION_FUNCTIONS.valid_keys()


def test_routes_sliding_layers_to_flash_attention(monkeypatch):
    calls = []

    def fake_flash(*args, **kwargs):
        calls.append(("flash_attention_2", kwargs.get("sliding_window")))
        query = args[1]
        return query.transpose(1, 2).contiguous(), None

    def fake_sdpa(*args, **kwargs):
        calls.append(("sdpa", kwargs.get("sliding_window")))
        query = args[1]
        return query.transpose(1, 2).contiguous(), None

    def get_attention_forward(name):
        return fake_flash if name == "flash_attention_2" else fake_sdpa

    monkeypatch.setattr(mixed, "_get_attention_forward", get_attention_forward)

    query = torch.randn(1, 2, 8, 256)
    key = torch.randn(1, 2, 8, 256)
    value = torch.randn(1, 2, 8, 256)
    module = DummyGemma4Attention(is_sliding=True)

    output, _ = mixed.gemma4_mixed_attention_forward(module, query, key, value, None, sliding_window=512)

    assert output.shape == (1, 8, 2, 256)
    assert calls == [("flash_attention_2", 512)]


def test_routes_global_layers_to_sdpa(monkeypatch):
    calls = []

    def fake_sdpa(*args, **kwargs):
        calls.append(("sdpa", kwargs.get("is_causal")))
        query = args[1]
        return query.transpose(1, 2).contiguous(), None

    monkeypatch.setattr(mixed, "_get_attention_forward", lambda name: fake_sdpa)
    monkeypatch.delenv(mixed.GLOBAL_BACKEND_ENV, raising=False)

    query = torch.randn(1, 2, 8, 512)
    key = torch.randn(1, 1, 8, 512)
    value = torch.randn(1, 1, 8, 512)
    module = DummyGemma4Attention(is_sliding=False)

    output, _ = mixed.gemma4_mixed_attention_forward(module, query, key, value, None)

    assert output.shape == (1, 8, 2, 512)
    assert calls == [("sdpa", None)]


def test_packed_position_ids_build_block_diagonal_causal_mask():
    query = torch.randn(1, 1, 5, 512)
    key = torch.randn(1, 1, 5, 512)
    position_ids = torch.tensor([[0, 1, 0, 1, 2]])

    mask = mixed._build_causal_or_packed_mask(
        query,
        key,
        None,
        sliding_window=None,
        position_ids=position_ids,
    )

    assert mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, 1, 0]
    assert not mask[0, 0, 2, 1]
    assert mask[0, 0, 4, 2]
    assert not mask[0, 0, 0, 1]


def test_sliding_fallback_mask_limits_context_window():
    query = torch.randn(1, 1, 5, 256)
    key = torch.randn(1, 1, 5, 256)
    position_ids = torch.tensor([[0, 1, 2, 3, 4]])

    mask = mixed._build_causal_or_packed_mask(
        query,
        key,
        None,
        sliding_window=2,
        position_ids=position_ids,
    )

    assert mask[0, 0, 4, 3]
    assert mask[0, 0, 4, 4]
    assert not mask[0, 0, 4, 2]
