"""Tests for supreme/scores.py — s_mcm, s_mmp, s_gmp."""

import torch
import pytest

from supreme.scores import s_mcm, s_mmp, s_gmp


B, C, D = 8, 5, 64  # batch, classes, embed dim


def rand_normalized(*shape):
    """Return L2-normalized random tensor."""
    x = torch.randn(*shape)
    return torch.nn.functional.normalize(x, dim=-1)


class TestSMcm:
    def test_output_shape(self):
        query = rand_normalized(B, D)
        protos = rand_normalized(C, D)
        out = s_mcm(query, protos, tau=0.01)
        assert out.shape == (B,)

    def test_values_in_range(self):
        query = rand_normalized(B, D)
        protos = rand_normalized(C, D)
        out = s_mcm(query, protos, tau=0.01)
        assert (out >= 0).all() and (out <= 1).all()

    def test_perfect_match_gives_high_score(self):
        """A query identical to one prototype should get a near-1 score."""
        protos = rand_normalized(C, D)
        query = protos[:1].clone()   # exact match to class 0
        out = s_mcm(query, protos, tau=0.01)
        assert out[0].item() > 0.9

    def test_no_grad_propagation(self):
        query = rand_normalized(B, D)
        protos = rand_normalized(C, D)
        out = s_mcm(query, protos)
        assert out.requires_grad is False


class TestSMmp:
    def test_output_shape(self):
        img_emb = rand_normalized(B, D)
        txt_proto = rand_normalized(C, D)
        img_proto = rand_normalized(C, D)
        out = s_mmp(img_emb, txt_proto, img_proto, tau=0.01)
        assert out.shape == (B,)

    def test_values_in_range(self):
        img_emb = rand_normalized(B, D)
        txt_proto = rand_normalized(C, D)
        img_proto = rand_normalized(C, D)
        out = s_mmp(img_emb, txt_proto, img_proto, tau=0.01)
        assert (out >= 0).all() and (out <= 1).all()

    def test_is_average_of_two_mcm(self):
        img_emb = rand_normalized(B, D)
        txt_proto = rand_normalized(C, D)
        img_proto = rand_normalized(C, D)
        mmp = s_mmp(img_emb, txt_proto, img_proto, tau=0.05)
        expected = (s_mcm(img_emb, txt_proto, tau=0.05) + s_mcm(img_emb, img_proto, tau=0.05)) / 2
        assert torch.allclose(mmp, expected)


class TestSGmp:
    def test_output_shape(self):
        img_emb = rand_normalized(B, D)
        I_prime = rand_normalized(B, D)
        txt_proto = rand_normalized(C, D)
        img_proto = rand_normalized(C, D)
        out = s_gmp(img_emb, I_prime, txt_proto, img_proto, tau=0.01)
        assert out.shape == (B,)

    def test_values_in_range(self):
        img_emb = rand_normalized(B, D)
        I_prime = rand_normalized(B, D)
        txt_proto = rand_normalized(C, D)
        img_proto = rand_normalized(C, D)
        out = s_gmp(img_emb, I_prime, txt_proto, img_proto, tau=0.01)
        assert (out >= 0).all() and (out <= 1).all()

    def test_is_average_of_four_mcm(self):
        img_emb = rand_normalized(B, D)
        I_prime = rand_normalized(B, D)
        txt_proto = rand_normalized(C, D)
        img_proto = rand_normalized(C, D)
        tau = 0.05
        gmp = s_gmp(img_emb, I_prime, txt_proto, img_proto, tau=tau)
        expected = (
            s_mcm(img_emb, txt_proto, tau)
            + s_mcm(img_emb, img_proto, tau)
            + s_mcm(I_prime, txt_proto, tau)
            + s_mcm(I_prime, img_proto, tau)
        ) / 4
        assert torch.allclose(gmp, expected)
