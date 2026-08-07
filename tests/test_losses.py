"""Tests for the CFM and DSM losses (Eq. 31 / Eq. 54) and CFG label dropout."""

import pytest
import torch

from eps.losses import (
    ConditionalFlowMatchingLoss,
    DenoisingScoreMatchingLoss,
    build_loss,
    cfg_label_dropout,
    sample_time,
)
from eps.paths import CondOTScheduler, CosineScheduler, GaussianProbabilityPath

SCHEDULERS = [CondOTScheduler(), CosineScheduler()]


class _OracleVelocity:
    """The exact conditional velocity for a dataset consisting of one point z:
    then the conditional field IS the marginal field, so the CFM loss at this
    'model' must be exactly zero."""

    def __init__(self, path, z):
        self.path, self.z = path, z

    def __call__(self, x, t, y=None):
        return self.path.conditional_velocity(x, self.z.expand_as(x), t)


class _OracleScore:
    def __init__(self, path, z):
        self.path, self.z = path, z

    def __call__(self, x, t, y=None):
        return self.path.conditional_score(x, self.z.expand_as(x), t)


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_cfm_loss_zero_at_oracle(sched):
    torch.manual_seed(0)
    path = GaussianProbabilityPath(sched)
    z_point = torch.randn(1, 8, dtype=torch.float64)
    z = z_point.expand(256, 8)
    loss = ConditionalFlowMatchingLoss(path, t_min=0.0, t_max=0.99)
    value = loss(_OracleVelocity(path, z_point), z)
    assert value.item() < 1e-16


@pytest.mark.parametrize("sched", SCHEDULERS)
@pytest.mark.parametrize("weighting", ["uniform", "ddpm"])
def test_dsm_loss_zero_at_oracle(sched, weighting):
    torch.manual_seed(1)
    path = GaussianProbabilityPath(sched)
    z_point = torch.randn(1, 8, dtype=torch.float64)
    z = z_point.expand(256, 8)
    loss = DenoisingScoreMatchingLoss(path, t_min=0.0, t_max=0.95, weighting=weighting)
    value = loss(_OracleScore(path, z_point), z)
    assert value.item() < 1e-14


def test_cfm_loss_positive_and_differentiable():
    torch.manual_seed(2)
    path = GaussianProbabilityPath(CondOTScheduler())
    net = torch.nn.Linear(4, 4)

    def model(x, t, y=None):
        return net(x)

    z = torch.randn(64, 4)
    loss = ConditionalFlowMatchingLoss(path)(model, z)
    assert loss.item() > 0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_sample_time_range():
    torch.manual_seed(3)
    t = sample_time(10000, torch.device("cpu"), t_min=0.2, t_max=0.7)
    assert t.min() >= 0.2 and t.max() <= 0.7
    assert abs(t.mean().item() - 0.45) < 0.01


def test_cfg_label_dropout():
    torch.manual_seed(4)
    y = torch.randint(0, 1000, (10000,))
    assert torch.equal(cfg_label_dropout(y, 1000, 0.0), y)
    assert (cfg_label_dropout(y, 1000, 1.0) == 1000).all()
    dropped = cfg_label_dropout(y, 1000, 0.1)
    frac = (dropped == 1000).float().mean().item()
    assert 0.07 < frac < 0.13  # ~eta, allowing that some true labels equal none
    changed = dropped != y
    assert (dropped[changed] == 1000).all()  # dropout only ever writes the null token


def test_build_loss_factory():
    path = GaussianProbabilityPath(CondOTScheduler())
    assert isinstance(build_loss("velocity", path), ConditionalFlowMatchingLoss)
    assert isinstance(build_loss("score", path), DenoisingScoreMatchingLoss)
    with pytest.raises(ValueError):
        build_loss("noise", path)
