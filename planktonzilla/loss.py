"""
(c) Inria

Imbalance-aware loss functions for the planktonzilla classification pipeline.

The plankton dataset is heavily long-tailed (a few abundant classes dominate the
many rare ones), so plain cross-entropy biases the model toward head classes. This
module collects loss variants that counteract that imbalance via focal down-weighting,
label-distribution-aware margins, asymmetric positive/negative weighting, or balanced
softmax priors. Every loss subclasses :class:`AbstractHFLoss` so it can be passed to a
Hugging Face ``Trainer`` as ``compute_loss_func`` and consume the
``ImageClassifierOutputWithNoAttention`` produced by the model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


class AbstractHFLoss(nn.Module):
    """Base class for the imbalance-aware losses used with the Hugging Face ``Trainer``.

    Concrete subclasses implement :meth:`forward`, which receives an
    ``ImageClassifierOutputWithNoAttention`` (the model output, whose ``.logits`` hold the
    raw class scores) and the integer ``target`` labels, and returns a scalar loss tensor.
    Instances are passed to ``Trainer(compute_loss_func=...)``.
    """

    def __init__(self):
        super().__init__()

    def forward(self, output: ImageClassifierOutputWithNoAttention, target, **kwargs):
        """Compute the loss for a model output and target.

        Subclasses must override this method and return a scalar loss tensor. The base
        implementation raises ``NotImplementedError``.
        """
        raise NotImplementedError("Not implemented!")


class FocalLoss(AbstractHFLoss):
    """Focal loss for class imbalance.

    Down-weights well-classified (high-confidence) examples by the factor
    ``(1 - pt) ** gamma`` so that training focuses on the hard, typically rare-class
    examples instead of being swamped by the easy head-class majority.

    Args:
        alpha: Per-class weighting. A scalar is expanded to the two-class vector
            ``[alpha, 1 - alpha]``; a list/tensor is used as explicit per-class weights.
        gamma: Focusing exponent — larger values down-weight easy examples more aggressively.
        size_average: If True, return the mean loss over the batch; otherwise the sum.

    *Source:* Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2018).
    **Focal Loss for Dense Object Detection.** arXiv preprint arXiv:1708.02002.
    <https://arxiv.org/abs/1708.02002>

    *Note:* Based on code from <https://github.com/clcarwin/focal_loss_pytorch>.
    """

    def __init__(self, alpha: float | int | list | torch.Tensor, gamma: float = 3, size_average: bool = True):
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

    Enforces larger classification margins for rare classes than for frequent ones: the
    per-class margin scales as ``1 / n_c**0.25`` (``n_c`` is the class count), so minority
    classes are pushed further from the decision boundary, improving their generalization.

    Args:
        cls_num_list: Number of training examples per class, used to derive the margins.
        max_m: Maximum margin; margins are rescaled so the largest equals this value.
        weight: Optional per-class weights passed through to ``cross_entropy`` (e.g. for DRW).
        s: Logit scaling factor applied before cross-entropy.

    *Source:*  Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Aréchiga, and Tengyu Ma.
    (2019). **Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss.** CoRR, vol. abs/1906.07413.
    <https://arxiv.org/abs/1906.07413>

    *Note:* Adapted from from: <https://github.com/kaidic/LDAM-DRW/blob/master/losses.py>.
    """

    def __init__(self, cls_num_list: list[int], max_m: float = 0.5, weight=None, s: int = 30):
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
    """Maximum Margin loss for imbalanced classification.

    Extends the LDAM idea by replacing the fixed label-distribution margin with a
    data-dependent margin estimated per batch from the gap between the positive (ground-truth)
    score and the strongest negative score (see :meth:`obj_margins`). With ``ldam=True`` the
    class-frequency margins are additionally blended in.

    Args:
        cls_num_list: Number of training examples per class, used to derive the base margins.
        max_m: Maximum margin scale.
        weight: Optional per-class weights passed to ``cross_entropy``.
        s: Logit scaling factor applied before cross-entropy.
        gamma: Sharpness of the foreground (positive-class) margin term in :meth:`obj_margins`.
        ldam: If True, subtract the class-frequency margin from ``max_m`` to combine the
            LDAM and maximum-margin schemes.

    *Source:* Kang, H., Vu, T., & Yoo, C. D. (2021). *Learning imbalanced datasets with maximum
    margin loss*. 2021 IEEE International Conference on Image Processing (ICIP), 1269-1273. IEEE.
    <https://arxiv.org/abs/2206.05380>

    *Note:* Code adapted from <https://github.com/ihaeyong/Maximum-Margin-LDAM>. This class
    subclasses ``nn.Module`` directly (not :class:`AbstractHFLoss`); its ``forward`` matches the
    same ``(output, target)`` interface so it is still ``Trainer``-compatible.
    """

    def __init__(
        self, cls_num_list: list[int], max_m: float = 0.5, weight=None, s: int = 30, gamma: float = 1.1, ldam: bool = False
    ):
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
    """Asymmetric loss (ASL) for imbalanced classification.

    Decouples the focusing applied to positive vs. negative samples so that the abundant
    easy negatives (dominant in long-tailed/multi-label settings) are down-weighted more
    strongly than positives, preventing them from overwhelming the gradient. Optional label
    smoothing (``eps``) further regularizes the targets.

    Args:
        gamma_pos: Focusing exponent for positive (ground-truth) classes.
        gamma_neg: Focusing exponent for negative classes; typically ``gamma_neg > gamma_pos``.
        eps: Label-smoothing strength; ``0`` disables smoothing.
        reduction: ``"mean"`` to average over the batch, otherwise the per-example loss is summed
            over classes and returned unreduced across the batch.

    *Source:* Emanuel Ben Baruch, Tal Ridnik, Nadav Zamir, Asaf Noy, Itamar Friedman, Matan Protter, and Lihi Zelnik-Manor.
    (2020). **Asymmetric Loss For Multi-Label Classification.** CoRR, vol. abs/2009.14119. <https://arxiv.org/abs/2009.14119>

    *Note:* Based on code from: <https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py>.
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction="mean"):
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
    """Robust Asymmetric Loss (RAL) for long-tailed multi-label learning.

    A robustified variant of :class:`AsymmetricLoss` that reshapes the positive and negative
    weighting with Taylor-style correction terms (``epsilon_pos``, ``epsilon_pos_pow``,
    ``epsilon_neg``) and a ``lamb`` factor, making the loss less sensitive to noisy/uncertain
    predictions on the long tail.

    Args:
        gamma_pos: Focusing exponent for positive classes.
        gamma_neg: Focusing exponent for negative classes.
        eps: Label-smoothing strength and probability floor used when clamping log-probabilities.
        epsilon_pos_pow: Coefficient of the second-order (squared) positive correction term.
        reduction: ``"mean"`` to average over the batch, otherwise summed over classes per example.

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
    """Balanced Meta-Softmax (BALMS) loss.

    Corrects the train/test label-distribution shift of long-tailed data by adding the log of
    each class's training frequency to its logit before softmax cross-entropy. This logit
    adjustment makes the softmax an unbiased estimator under a balanced test distribution,
    favoring rare classes without explicit resampling.

    Args:
        cls_num_list: Number of training examples per class; its log is used as the prior offset.

    *Source:* Jiawei Ren, Cunjun Yu, Shunan Sheng, Xiao Ma, Haiyu Zhao, Shuai Yi, and Hongsheng Li.
    (2020). **Balanced Meta-Softmax for Long-Tailed Visual Recognition.** NeurIPS 2020.
    <https://arxiv.org/abs/2007.10740>
    """

    def __init__(self, cls_num_list: list[int]):
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
    """Standard cross-entropy loss wrapped for the Hugging Face ``Trainer``.

    Baseline loss with no imbalance correction beyond the optional per-class ``weight``. It
    exists so plain cross-entropy can be selected through the same config/dispatch path as the
    imbalance-aware losses.

    Args:
        weight: Optional per-class rescaling weights passed to ``cross_entropy``.
    """

    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, output, target, **kwargs):
        """Compute weighted cross-entropy over ``output.logits`` against ``target``."""
        return F.cross_entropy(output.logits, target, weight=self.weight)
