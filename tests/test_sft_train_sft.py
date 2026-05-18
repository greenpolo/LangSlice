from __future__ import annotations

import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock


class _FakeDataset:
    examples = []


def test_train_passes_max_seq_length_to_unsloth_and_collator(monkeypatch, tmp_path):
    from sft import train_sft

    fake_model = MagicMock(name="model")
    fake_processor = MagicMock(name="processor")
    fake_fast_vision = MagicMock()
    fake_fast_vision.from_pretrained.return_value = (fake_model, fake_processor)
    fake_fast_vision.get_peft_model.return_value = fake_model

    fake_unsloth = types.SimpleNamespace(FastVisionModel=fake_fast_vision)
    fake_trainer = MagicMock()
    fake_trl = types.SimpleNamespace(
        SFTConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        SFTTrainer=MagicMock(return_value=fake_trainer),
    )
    fake_rlvr_atlas_grid = types.SimpleNamespace(build_atlas_grid=lambda pairs: {})
    fake_rlvr = types.SimpleNamespace(atlas_grid=fake_rlvr_atlas_grid)

    class FakeCollator:
        def __init__(
            self,
            *,
            processor,
            max_seq_length,
            atlas_cache=None,
            query_cache=None,
            enable_splice=False,
        ):
            self.processor = processor
            self.max_seq_length = max_seq_length
            self.atlas_cache = atlas_cache
            self.query_cache = query_cache
            self.enable_splice = enable_splice

    class FakeCallback:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    monkeypatch.setitem(sys.modules, "rlvr", fake_rlvr)
    monkeypatch.setitem(sys.modules, "rlvr.atlas_grid", fake_rlvr_atlas_grid)
    monkeypatch.setattr(train_sft, "LangSliceCollator", FakeCollator)
    monkeypatch.setattr(
        "sft.eval.BaselineEvalCallback",
        FakeCallback,
        raising=False,
    )
    monkeypatch.setattr(
        "sft.eval.AgentLoopEvalCallback",
        FakeCallback,
        raising=False,
    )

    args = Namespace(
        output_dir=tmp_path,
        test_images_root=Path("references/TestImages"),
        report_to="none",
    )
    config = {
        "sft": {
            "base_model": "unsloth/gemma-4-E4B-it",
            "load_in_4bit": True,
            "max_seq_length": 8192,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "report_to": "none",
        },
        "lora": {
            "finetune_vision_layers": False,
            "finetune_language_layers": True,
            "finetune_attention_modules": True,
            "finetune_mlp_modules": True,
            "r": 8,
            "lora_alpha": 16,
            "use_gradient_checkpointing": "unsloth",
        },
    }

    train_sft._train(args, config, _FakeDataset(), _FakeDataset(), MagicMock(), seed=123)

    assert fake_fast_vision.from_pretrained.call_args.kwargs["max_seq_length"] == 8192
    assert fake_fast_vision.get_peft_model.call_args.kwargs["max_seq_length"] == 8192
    trainer_kwargs = fake_trl.SFTTrainer.call_args.kwargs
    assert trainer_kwargs["data_collator"].max_seq_length == 8192
    assert fake_trainer.data_collator is trainer_kwargs["data_collator"]
    fake_trainer.train.assert_called_once()
