"""Tests for supreme/losses.py — individual losses and total_loss."""

import torch
import torch.nn.functional as F
import pytest

from supreme.config import Config
from supreme.losses import cosine_logits, l_bias, l_id, l_inter, l_intra, total_loss


B, C, D, N = 8, 5, 64, 32  # batch, classes, embed_dim, n_lm


def rand_norm(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)


def make_fwd_dict():
    """Build a synthetic forward-pass dictionary matching SUPREME.forward() output."""
    img_emb = rand_norm(B, D)
    txt_proto = rand_norm(C, D)
    I_prime = rand_norm(B, D)
    I_hat = rand_norm(B, D)
    P_hat = rand_norm(C, D)
    P_img_sp = rand_norm(C, D)
    b = torch.randn(B, N)
    m_I = torch.randn(B, N)
    labels = torch.randint(0, C, (B,))
    return dict(
        img_emb=img_emb,
        txt_proto=txt_proto,
        I_prime=I_prime,
        I_hat=I_hat,
        P_hat=P_hat,
        P_img_sp=P_img_sp,
        b=b,
        m_I=m_I,
        labels=labels,
    )


class TestCosineLogits:
    def test_output_shape(self):
        a = rand_norm(B, D)
        b = rand_norm(C, D)
        out = cosine_logits(a, b, tau=0.01)
        assert out.shape == (B, C)

    def test_scaling_by_tau(self):
        a = rand_norm(B, D)
        b = rand_norm(C, D)
        out1 = cosine_logits(a, b, tau=0.01)
        out2 = cosine_logits(a, b, tau=0.1)
        assert torch.allclose(out1, out2 * 10, atol=1e-5)


class TestLId:
    def test_returns_scalar(self):
        img_emb = rand_norm(B, D)
        txt_proto = rand_norm(C, D)
        labels = torch.randint(0, C, (B,))
        loss = l_id(img_emb, txt_proto, labels, tau=0.01)
        assert loss.shape == ()

    def test_non_negative(self):
        img_emb = rand_norm(B, D)
        txt_proto = rand_norm(C, D)
        labels = torch.randint(0, C, (B,))
        loss = l_id(img_emb, txt_proto, labels, tau=0.01)
        assert loss.item() >= 0

    def test_perfect_alignment_low_loss(self):
        """Embeddings identical to their class prototype should give low loss."""
        txt_proto = rand_norm(C, D)
        labels = torch.arange(B) % C
        img_emb = txt_proto[labels]   # each image = its own class prototype
        loss = l_id(img_emb, txt_proto, labels, tau=0.01)
        assert loss.item() < 0.1


class TestLInter:
    def test_returns_scalar(self):
        fwd = make_fwd_dict()
        loss = l_inter(
            fwd["img_emb"], fwd["txt_proto"],
            fwd["I_prime"], fwd["P_img_sp"],
            fwd["labels"], tau=0.01,
        )
        assert loss.shape == ()

    def test_non_negative(self):
        fwd = make_fwd_dict()
        loss = l_inter(
            fwd["img_emb"], fwd["txt_proto"],
            fwd["I_prime"], fwd["P_img_sp"],
            fwd["labels"], tau=0.01,
        )
        assert loss.item() >= 0


class TestLIntra:
    def test_returns_scalar(self):
        fwd = make_fwd_dict()
        loss = l_intra(
            fwd["img_emb"], fwd["txt_proto"],
            fwd["I_hat"], fwd["P_hat"],
        )
        assert loss.shape == ()

    def test_zero_when_perfect_reconstruction(self):
        img_emb = rand_norm(B, D)
        txt_proto = rand_norm(C, D)
        loss = l_intra(img_emb, txt_proto, img_emb.clone(), txt_proto.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self):
        fwd = make_fwd_dict()
        loss = l_intra(
            fwd["img_emb"], fwd["txt_proto"],
            fwd["I_hat"], fwd["P_hat"],
        )
        assert loss.item() >= 0


class TestLBias:
    def test_returns_scalar(self):
        mu = torch.zeros(N)
        b = torch.randn(B, N)
        m_I = torch.randn(B, N)
        loss = l_bias(mu, b, m_I)
        assert loss.shape == ()

    def test_zero_when_aligned(self):
        mu = torch.ones(N)
        m_I = torch.ones(B, N)
        b = torch.ones(B, N)
        loss = l_bias(mu, b, m_I)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_non_negative(self):
        mu = torch.randn(N)
        b = torch.randn(B, N)
        m_I = torch.randn(B, N)
        loss = l_bias(mu, b, m_I)
        assert loss.item() >= 0


class TestTotalLoss:
    def test_returns_tensor_and_dict(self):
        cfg = Config()
        fwd = make_fwd_dict()
        mu = torch.zeros(N)
        total, components = total_loss(fwd, cfg, mu)
        assert isinstance(total, torch.Tensor)
        assert isinstance(components, dict)

    def test_components_keys(self):
        cfg = Config()
        fwd = make_fwd_dict()
        mu = torch.zeros(N)
        _, components = total_loss(fwd, cfg, mu)
        expected_keys = {"loss_id", "loss_inter", "loss_intra", "loss_bias", "loss_total"}
        assert set(components.keys()) == expected_keys

    def test_component_values_are_floats(self):
        cfg = Config()
        fwd = make_fwd_dict()
        mu = torch.zeros(N)
        _, components = total_loss(fwd, cfg, mu)
        assert all(isinstance(v, float) for v in components.values())

    def test_total_is_non_negative(self):
        cfg = Config()
        fwd = make_fwd_dict()
        mu = torch.zeros(N)
        total, _ = total_loss(fwd, cfg, mu)
        assert total.item() >= 0

    def test_gradients_flow(self):
        cfg = Config()
        fwd = make_fwd_dict()
        for key in ("img_emb", "txt_proto", "I_prime", "I_hat", "P_hat", "P_img_sp"):
            fwd[key].requires_grad_(True)
        mu = torch.zeros(N, requires_grad=True)
        total, _ = total_loss(fwd, cfg, mu)
        total.backward()
        assert mu.grad is not None
