"""Tests for EMA and the config system."""

import torch
import torch.nn as nn

from eps.config import EpsilonConfig, apply_overrides, from_dict, load_config, to_dict
from eps.training.ema import EMA


def _net():
    torch.manual_seed(0)
    return nn.Linear(4, 4)


def test_ema_initial_copy():
    net = _net()
    ema = EMA(net, decay=0.999)
    for name, p in net.named_parameters():
        assert torch.equal(ema.shadow[name], p)


def test_ema_single_update_math():
    net = _net()
    ema = EMA(net, decay=0.5, warmup=False)
    old = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)
    ema.update(net)
    for name, p in net.named_parameters():
        expected = 0.5 * old[name] + 0.5 * p
        assert torch.allclose(ema.shadow[name], expected)


def test_ema_warmup_ramps():
    net = _net()
    ema = EMA(net, decay=0.9999, warmup=True)
    ema.step = 0
    assert ema.current_decay() < 0.2  # (1)/(10) at the first update
    ema.step = 100000
    assert abs(ema.current_decay() - 0.9999) < 1e-6


def test_ema_store_copy_restore():
    net = _net()
    ema = EMA(net, decay=0.0, warmup=False)  # shadow == live at all times
    live = {k: v.clone() for k, v in net.named_parameters()}
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(2.0)
    ema.update(net)  # decay 0 -> shadow = live*2
    ema.store(net)
    ema.copy_to(net)
    ema.restore(net)
    for name, p in net.named_parameters():
        assert torch.allclose(p, live[name] * 2.0)


def test_ema_state_dict_roundtrip():
    net = _net()
    ema = EMA(net, decay=0.99)
    ema.update(net)
    state = ema.state_dict()
    ema2 = EMA(net, decay=0.5)
    ema2.load_state_dict(state)
    assert ema2.decay == 0.99 and ema2.step == 1
    for k in ema.shadow:
        assert torch.equal(ema.shadow[k], ema2.shadow[k])


def test_config_defaults_and_roundtrip():
    cfg = EpsilonConfig()
    assert cfg.path.scheduler == "condot"
    assert cfg.model.prediction == "velocity"
    d = to_dict(cfg)
    cfg2 = from_dict(d)
    assert to_dict(cfg2) == d


def test_config_overrides():
    cfg = EpsilonConfig()
    cfg = apply_overrides(
        cfg,
        ["training.lr=2e-4", "model.name=dit", "model.unet.channel_mult=[1,2,4]",
         "logging.wandb=true"],
    )
    assert cfg.training.lr == 2e-4
    assert cfg.model.name == "dit"
    assert cfg.model.unet.channel_mult == (1, 2, 4)
    assert cfg.logging.wandb is True


def test_config_rejects_unknown_keys():
    import pytest

    with pytest.raises(KeyError):
        from_dict({"nope": {}})
    with pytest.raises(KeyError):
        apply_overrides(EpsilonConfig(), ["training.nope=1"])


def test_load_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("training:\n  lr: 3.0e-4\nmodel:\n  name: dit\n")
    cfg = load_config(p, overrides=["training.batch_size=16"])
    assert cfg.training.lr == 3e-4
    assert cfg.model.name == "dit"
    assert cfg.training.batch_size == 16
