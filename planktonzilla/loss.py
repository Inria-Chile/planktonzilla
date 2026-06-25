"""
(c) Inria

Imbalance-aware loss functions for plankton image classification.

Collects loss functions tailored to long-tailed / class-imbalanced datasets
(focal, LDAM, maximum-margin, asymmetric, robust-asymmetric, balanced
meta-softmax) plus a plain cross-entropy baseline. Every loss derives from
`AbstractHFLoss` and exposes a `forward(output, target)` signature compatible
with the Hugging Face `Trainer` custom-loss hook, where ``output`` is an
`ImageClassifierOutputWithNoAttention` carrying the model logits.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


class AbstractHFLoss(nn.Module):
    """Base class for losses compatible with the Hugging Face transformers Trainer.

    Subclasses implement `forward`, taking the model output and target labels and
    returning a scalar loss tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs):
        """Compute the loss for a model output and target.

        Subclasses must implement this method and return a scalar loss tensor.
        """
        raise NotImplementedError("Not implemented!")


class FocalLoss(AbstractHFLoss):
    """Focal loss.

    *Source:* Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2018).
    **Focal Loss for Dense Object Detection.** arXiv preprint arXiv:1708.02002.
    <https://arxiv.org/abs/1708.02002>

    *Note:* Based on code from <https://github.com/clcarwin/focal_loss_pytorch>.
    """

    def __init__(self, alpha: float | int | list | torch.Tensor, gamma: float = 3, size_average: bool = True):
        """Configure the focal loss.

        Args:
            alpha: Class-balancing weight(s). A scalar is expanded to the
                two-class form ``[alpha, 1 - alpha]``; a list/tensor is used as a
                per-class weight vector.
            gamma: Focusing parameter; higher values down-weight easy, well-
                classified examples more strongly.
            size_average: When ``True`` return the mean loss over the batch,
                otherwise the sum.
        """
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs):
        """Compute focal loss given model `output` and `target` labels.

        Returns a scalar loss (mean or sum depending on `size_average`).
        """
        logits = output.logits

        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1, logits.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)

        logpt = F.log_softmax(logits, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != logits.data.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LDAMLoss(AbstractHFLoss):
    """Label-Distribution-Aware Margin (LDAM) loss.

    *Source:*  Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Aréchiga, and Tengyu Ma.
    (2019). **Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss.** CoRR, vol. abs/1906.07413.
    <https://arxiv.org/abs/1906.07413>

    *Note:* Adapted from from: <https://github.com/kaidic/LDAM-DRW/blob/master/losses.py>.
    """

    def __init__(self, cls_num_list: list[int], max_m: float = 0.5, weight=None, s: int = 30):
        """Configure the LDAM loss.

        Args:
            cls_num_list: Number of training samples per class; used to derive
                the per-class margins (rarer classes get larger margins).
            max_m: Maximum margin; the per-class margins are scaled so the
                largest equals this value.
            weight: Optional per-class weight tensor passed to the underlying
                cross-entropy.
            s: Positive logit scaling factor applied before cross-entropy.
        """
        super().__init__()
        assert cls_num_list is not None
        assert s > 0

        self.cls_num_list = cls_num_list
        self.max_m = max_m
        self.weight = weight
        self.s = s

        m_list = 1.0 / np.sqrt(np.sqrt(self.cls_num_list))
        m_list = m_list * (self.max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list

    def forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs):
        """Compute LDAM loss using class margin adjustments.

        Applies label-distribution-aware margins before computing cross-entropy.
        """
        logits = output.logits

        device = logits.device
        self.m_list = self.m_list.to(device)
        index = torch.zeros_like(logits, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor).to(device=target.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = logits - batch_m

        output = torch.where(index, x_m, logits)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class MaximumMarginLoss(nn.Module):
    """Maximum Margin loss.

    *Source:* Kang, H., Vu, T., & Yoo, C. D. (2021). *Learning imbalanced datasets with maximum
    margin loss*. 2021 IEEE International Conference on Image Processing (ICIP), 1269-1273. IEEE.
    <https://arxiv.org/abs/2206.05380>

    *Note:* Code adapted from <https://github.com/ihaeyong/Maximum-Margin-LDAM>.
    """

    def __init__(
        self, cls_num_list: list[int], max_m: float = 0.5, weight=None, s: int = 30, gamma: float = 1.1, ldam: bool = False
    ):
        """Configure the maximum-margin loss.

        Args:
            cls_num_list: Number of training samples per class; used to derive the
                per-class margins.
            max_m: Base maximum margin.
            weight: Optional per-class weight tensor passed to cross-entropy.
            s: Positive logit scaling factor applied before cross-entropy.
            gamma: Exponential decay factor controlling the object-margin term.
            ldam: When ``True`` combine the maximum margin with the
                LDAM-style per-class margin instead of using a fixed ``max_m``.
        """
        super().__init__()

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (0.5 / np.max(m_list))
        m_list = torch.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.max_m = max_m
        self.gamma = gamma
        self.ldam = ldam

    def weight(self, freq_bias, target, args):
        """Compute per-class weights from frequency bias and `args.beta`.

        Returns a tensor with a weight per class to rebalance losses.
        """

        index = torch.zeros_like(freq_bias, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor)

        # plus 1 affects top-1 acc.
        cls_num_list = index_float.sum(0).data.cpu() + 1

        beta = args.beta

        effect_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effect_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights)  # .cuda(args.gpu)

        return per_cls_weights

    def obj_margins(self, rm_obj_dists, labels, index_float, max_m):
        """Estimate object margins between positive and negative distances.

        Used internally to compute per-example margin adjustments for the
        maximum-margin objective.
        """

        obj_neg_labels = 1.0 - index_float
        obj_neg_dists = rm_obj_dists * obj_neg_labels

        min_pos_prob = rm_obj_dists[:, labels.data.cpu().numpy()[0]].data
        max_neg_prob = obj_neg_dists.max(1)[0].data

        # estimate the margin between dists and gt labels
        batch_m_fg = torch.max(min_pos_prob - max_neg_prob, torch.zeros_like(min_pos_prob))[:, None]

        mask_fg = (batch_m_fg > 0).float()
        batch_fg = torch.exp(-batch_m_fg - max_m * self.gamma) * mask_fg

        batch_m_bg = torch.max(max_neg_prob - min_pos_prob, torch.zeros_like(max_neg_prob))[:, None]

        mask_ng = (batch_m_bg > 0).float()
        batch_ng = torch.exp(-batch_m_bg - max_m) * mask_ng
        batch_m = batch_ng + batch_fg

        return batch_m.data

    def forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs):
        """Compute maximum-margin loss.

        Applies per-class/object margins and returns cross-entropy over
        adjusted logits.
        """
        x = output.logits
        self.m_list = self.m_list.to(target.device)
        index = torch.zeros_like(x, dtype=torch.uint8).to(target.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor).to(target.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))

        # 1.0 - [0.5] => [0.0 ~ 0.5]
        if self.ldam:
            max_m = self.max_m - batch_m
        else:
            max_m = self.max_m

        with torch.no_grad():
            batch_hmm = self.obj_margins(x, target, index_float, max_m)

        x_m = x - batch_hmm

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


class AsymmetricLoss(AbstractHFLoss):
    """Asymmetric loss

    *Source:* Emanuel Ben Baruch, Tal Ridnik, Nadav Zamir, Asaf Noy, Itamar Friedman, Matan Protter, and Lihi Zelnik-Manor.
    (2020). **Asymmetric Loss For Multi-Label Classification.** CoRR, vol. abs/2009.14119. <https://arxiv.org/abs/2009.14119>

    *Note:* Based on code from: <https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py>.
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction="mean"):
        """Configure the asymmetric loss.

        Args:
            gamma_pos: Focusing parameter applied to positive (target) classes.
            gamma_neg: Focusing parameter applied to negative classes; typically
                larger than ``gamma_pos`` to down-weight easy negatives more.
            eps: Label-smoothing factor; ``0`` disables smoothing.
            reduction: ``"mean"`` to average over the batch, anything else leaves
                the per-sample summed loss unreduced.
        """
        super().__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs):
        """Compute Asymmetric loss for (optionally) multi-label inputs.

        Implements label smoothing and asymmetric weighting for positive and
        negative classes before returning the reduced loss.
        """
        inputs = output.logits
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class RobustAsymmetricLoss(AbstractHFLoss):
    """Robust Asymmetric Loss (RAL)

    *Source:* Wongi Park, Inhyuk Park, Sungeun Kim, and Jongbin Ryu. (2023). **Robust Asymmetric Loss
    for Multi-Label Long-Tailed Learning.** arXiv preprint arXiv:2308.05542.
    <https://arxiv.org/abs/2308.05542>

    *Note:* Code based on <https://github.com/kalelpark/RAL/blob/main/models/get_optimizer.py>
    """

    def __init__(
        self,
        gamma_pos=0,
        gamma_neg=4,
        eps: float = 0.1,
        epsilon_pos_pow=-2.5,
        reduction="mean",
    ):
        """Configure the robust asymmetric loss.

        Args:
            gamma_pos: Focusing parameter applied to positive (target) classes.
            gamma_neg: Focusing parameter applied to negative classes.
            eps: Clamp/label-smoothing floor used for numerical stability and
                smoothing.
            epsilon_pos_pow: Coefficient of the second-order robustness term on
                the positive branch.
            reduction: ``"mean"`` to average over the batch, anything else leaves
                the per-sample summed loss unreduced.
        """
        super().__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.epsilon_pos = 1.0
        self.epsilon_neg = 0.0
        self.epsilon_pos_pow = epsilon_pos_pow
        self.lamb = 1.5

    def forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs):
        """Compute Robust Asymmetric Loss (RAL).

        Variant of Asymmetric loss with additional robustness terms.
        """
        inputs = output.logits
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = (
            torch.exp(log_preds)
            * (
                torch.log(xs_pos.clamp(min=self.eps))
                + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps))
                + self.epsilon_pos_pow * 0.5 * torch.pow(1 - xs_pos.clamp(min=self.eps), 2)
            )
            * torch.log(xs_pos)
        )
        xs_neg = (
            (1 - xs_pos)
            * (torch.log(xs_neg.clamp(min=self.eps)) + self.epsilon_neg * (xs_neg.clamp(min=self.eps)))
            * -(self.lamb - xs_neg)
            * xs_neg**2
        )
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == "mean":
            loss = loss.mean()

        return loss


class BalancedMetaSoftmaxLoss(AbstractHFLoss):
    """Balanced Meta-Softmax (BALMS) loss."""

    def __init__(self, cls_num_list: list[int]):
        """Configure the balanced meta-softmax loss.

        Args:
            cls_num_list: Number of training samples per class; their log values
                are added to the logits as priors to counteract class imbalance.
        """
        super().__init__()
        self.cls_num_list = torch.tensor(cls_num_list).float()

    def forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs):
        """Compute Balanced Meta-Softmax loss.

        Adjusts logits by log class priors before computing cross-entropy.
        """
        logits = output.logits
        adjusted_logits = logits + self.cls_num_list.log().to(logits.device)
        loss = F.cross_entropy(adjusted_logits, target)
        return loss


class CrossEntropyLossHF(AbstractHFLoss):
    """Plain cross-entropy loss, as a Hugging Face Trainer-compatible baseline.

    Wraps `torch.nn.functional.cross_entropy` so it can be dropped into the same
    custom-loss hook as the imbalance-aware losses in this module.
    """

    def __init__(self, weight=None):
        """Configure the cross-entropy loss.

        Args:
            weight: Optional per-class weight tensor passed straight through to
                `F.cross_entropy`.
        """
        super().__init__()
        self.weight = weight

    def forward(self, output, target, **kwargs):
        """Compute weighted cross-entropy over the model logits and targets."""
        return F.cross_entropy(output.logits, target, weight=self.weight)
