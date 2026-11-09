# loss.py
# Supervised Contrastive Loss (SupCon)
# Used on projected embeddings u = normalize(projector(encoder(x)))

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Inputs:
      feats  : (B, D) projected features (can be unnormalized; we normalize inside)
      labels : (B,)   integer class labels
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, feats, labels):
        # feats: (B,D), labels: (B,)
        feats = F.normalize(feats, dim=1)               # unit-norm embeddings
        B = feats.size(0)

        # similarity matrix (B,B)
        sim = torch.matmul(feats, feats.t()) / self.tau

        # mask: positives if same label, excluding self-pairs
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float()
        self_mask = torch.eye(B, device=feats.device)
        pos_mask = pos_mask * (1.0 - self_mask)         # remove diagonal

        # log-softmax over columns (for each anchor i, compare against all j)
        log_prob = sim - torch.logsumexp(sim - 1e9 * self_mask, dim=1, keepdim=True)
        # Explanation of the line above:
        # - we subtract a huge number on diagonal before logsumexp so "self" doesn't participate

        # for each anchor: average log-prob over positives
        pos_count = pos_mask.sum(dim=1)                  # (B,)
        # avoid division by zero: if a class appears once in batch, it has no positives
        pos_count = torch.clamp(pos_count, min=1.0)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count

        # loss = - average over batch
        loss = -mean_log_prob_pos.mean()
        return loss
