"""Tests for the model backbones: shapes, conditioning, AdaLN-Zero identity
at init, gradient checkpointing, VAE, and embeddings."""

import math

import pytest
import torch

from eps.models.dit import DiT
from eps.models.embeddings import FourierTimeEmbedding, sincos_pos_embed_2d
from eps.models.unet import UNet
from eps.models.vae import VAE, VAEOutput, vae_loss


def small_unet(**kw):
    args = dict(
        image_size=32,
        in_channels=3,
        model_channels=32,
        channel_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        num_heads=4,
        dropout=0.0,
        num_classes=10,
    )
    args.update(kw)
    return UNet(**args)


def small_dit(**kw):
    args = dict(
        input_size=16,
        patch_size=4,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=10,
    )
    args.update(kw)
    return DiT(**args)


def test_fourier_time_embedding_is_normed():
    """Eq. (68): ||TimeEmb(t)|| = 1 for every t."""
    emb = FourierTimeEmbedding(dim=64)
    t = torch.rand(128)
    out = emb(t)
    assert out.shape == (128, 64)
    assert torch.allclose(out.norm(dim=1), torch.ones(128), atol=1e-5)


def test_fourier_frequencies_geometric():
    """Eq. (69): w_1 = w_min, w_{d/2} = w_max, geometric in between."""
    emb = FourierTimeEmbedding(dim=8, w_min=0.5, w_max=32.0)
    f = emb.freqs
    assert math.isclose(f[0].item(), 0.5, rel_tol=1e-6)
    assert math.isclose(f[-1].item(), 32.0, rel_tol=1e-6)
    ratios = (f[1:] / f[:-1])
    assert torch.allclose(ratios, ratios[0].expand_as(ratios), atol=1e-5)


def test_sincos_pos_embed_shape():
    pe = sincos_pos_embed_2d(64, 4)
    assert pe.shape == (16, 64)


@pytest.mark.parametrize("with_labels", [False, True])
def test_unet_forward_shape(with_labels):
    torch.manual_seed(0)
    net = small_unet()
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,)) if with_labels else None
    out = net(x, t, y)
    assert out.shape == x.shape


def test_unet_zero_init_output():
    """The final conv is zero-initialised: the untrained net outputs 0,
    so early training starts from the identity-free baseline."""
    net = small_unet()
    x = torch.randn(2, 3, 32, 32)
    out = net(x, torch.rand(2), None)
    assert torch.allclose(out, torch.zeros_like(out))


def test_unet_null_label_differs_from_class_label():
    torch.manual_seed(1)
    net = small_unet()
    # Perturb every parameter: at init all residual branches end in
    # zero-initialised convs (by design), so conditioning cannot reach the
    # output until the weights move off zero.
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(2, 3, 32, 32)
    t = torch.full((2,), 0.5)
    out_y = net(x, t, torch.tensor([3, 3]))
    out_null = net(x, t, None)
    assert not torch.allclose(out_y, out_null)


@pytest.mark.parametrize("with_labels", [False, True])
def test_dit_forward_shape(with_labels):
    torch.manual_seed(2)
    net = small_dit()
    x = torch.randn(2, 3, 16, 16)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,)) if with_labels else None
    out = net(x, t, y)
    assert out.shape == x.shape


def test_dit_zero_init_output():
    net = small_dit()
    x = torch.randn(2, 3, 16, 16)
    out = net(x, torch.rand(2), None)
    assert torch.allclose(out, torch.zeros_like(out))


def test_dit_cross_attention_text_conditioning():
    torch.manual_seed(3)
    net = small_dit(cross_attention=True, context_dim=32)
    x = torch.randn(2, 3, 16, 16)
    t = torch.rand(2)
    context = torch.randn(2, 7, 32)  # (B, S, context_dim) text tokens
    out = net(x, t, context)
    assert out.shape == x.shape
    out_null = net(x, t, None)  # learned null context
    assert out_null.shape == x.shape


def test_dit_unpatchify_roundtrip():
    net = small_dit()
    b, p, g, c = 2, net.patch_size, net.grid_size, 3
    x = torch.randn(b, c, 16, 16)
    # patchify by hand (channel-major within patches, matching unpatchify)
    patches = x.reshape(b, c, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(b, g * g, p * p * c)
    assert torch.allclose(net.unpatchify(patches), x, atol=1e-6)


@pytest.mark.parametrize("factory", [small_unet, small_dit])
def test_gradient_checkpointing_matches(factory):
    torch.manual_seed(4)
    net = factory(gradient_checkpointing=False)
    net_ckpt = factory(gradient_checkpointing=True)
    net_ckpt.load_state_dict(net.state_dict())
    size = 32 if isinstance(net, UNet) else 16
    x = torch.randn(2, 3, size, size)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    net.train(), net_ckpt.train()
    a, b = net(x, t, y), net_ckpt(x, t, y)
    assert torch.allclose(a, b, atol=1e-6)
    (a.sum()).backward()
    (b.sum()).backward()
    for (n1, p1), (n2, p2) in zip(net.named_parameters(), net_ckpt.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5), n1


def test_vae_shapes_and_loss():
    torch.manual_seed(5)
    vae = VAE(in_channels=3, latent_channels=4, base_channels=16, channel_mult=(1, 2))
    x = torch.randn(2, 3, 32, 32)
    out = vae(x)
    assert out.reconstruction.shape == x.shape
    assert out.mean.shape == (2, 4, 16, 16)  # one downsample -> /2
    loss, logs = vae_loss(out, x, kl_weight=1e-6)
    assert torch.isfinite(loss)
    assert set(logs) == {"vae/recon", "vae/kl", "vae/loss"}


def test_vae_kl_zero_at_standard_normal():
    """KL(N(0,I) || N(0,I)) = 0 (Example 31)."""
    x = torch.zeros(2, 3, 8, 8)
    out = VAEOutput(
        reconstruction=x, mean=torch.zeros(2, 4, 4, 4), logvar=torch.zeros(2, 4, 4, 4)
    )
    _, logs = vae_loss(out, x, kl_weight=1.0)
    assert abs(logs["vae/kl"]) < 1e-12
    assert abs(logs["vae/recon"]) < 1e-12
