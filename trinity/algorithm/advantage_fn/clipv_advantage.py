"""GRPO advantage computation with Clip_V token filtering.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Tuple

import torch

if TYPE_CHECKING:
    from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn


class ClipVAdvantageFn(AdvantageFn):
    """Clip_V advantage: one-side clip only negative-advantage tokens,
    and cap the global clipped-token ratio."""

    def __init__(
        self,
        epsilon: float = 1e-6,
        mu: float = 2.0,
        max_frac: float = 1e-4,
    ) -> None:
        self.epsilon = epsilon
        self.mu = mu
        self.max_frac = max_frac

    def __call__(
        self,
        exps: "DataProto",
        **kwargs,
    ) -> Tuple["DataProto", Dict]:
        token_level_rewards = exps.batch["token_level_rewards"]
        response_mask = exps.batch["response_mask"]
        index = exps.non_tensor_batch["uid"]

        new_log_probs = exps.batch["old_log_probs"]
        new_entropys = exps.batch["entropys"]
        necs = exps.batch["necs"]

        response_length = token_level_rewards.shape[-1]
        scores = token_level_rewards.sum(dim=-1)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}
        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])

            for idx, grouped_scores in id2score.items():
                if len(grouped_scores) == 1:
                    id2mean[idx] = torch.tensor(0.0, dtype=scores.dtype, device=scores.device)
                    id2std[idx] = torch.tensor(1.0, dtype=scores.dtype, device=scores.device)
                elif len(grouped_scores) > 1:
                    group_scores = torch.stack(grouped_scores).to(
                        dtype=scores.dtype, device=scores.device
                    )
                    id2mean[idx] = torch.mean(group_scores)
                    id2std[idx] = torch.std(group_scores)
                else:
                    raise ValueError(f"no score in prompt index: {idx}")

            for i in range(bsz):
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + self.epsilon)
            scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

        exps.batch["advantages"] = scores
        exps.batch["returns"] = scores.clone()

        LP = new_log_probs
        H = new_entropys
        N = necs
        M = response_mask
        p = LP.exp()
        S = p * (H + LP)

        xD = (N - S).detach().to(torch.float32)
        A = exps.batch["advantages"].detach().to(torch.float32)
        m = M.to(torch.float32)

        n = m.sum().clamp_min(1.0)
        mean_d = (xD * m).sum() / n
        var_d = ((xD - mean_d) ** 2 * m).sum() / n
        std_d = var_d.sqrt()

        if std_d.item() < 1e-12:
            keep = torch.ones_like(M, dtype=M.dtype)
        else:
            pos_mask = A > 0
            neg_mask = A < 0

            keep_neg = xD <= (self.mu * std_d)
            keep_bool = torch.where(pos_mask, torch.ones_like(pos_mask), keep_neg)

            total_tokens = m.sum().clamp_min(1.0)
            clipped_mask = (M > 0) & (~keep_bool)
            frac_clipped = (clipped_mask.to(torch.float32).sum() / total_tokens).item()

            if frac_clipped <= self.max_frac:
                keep = keep_bool.to(M.dtype)
            else:
                max_clipped_tokens = max(int(self.max_frac * total_tokens.item()), 1)
                neg_to_clip = neg_mask & (M > 0) & (~keep_neg)
                neg_to_clip_count = int(neg_to_clip.to(torch.int32).sum().item())

                if neg_to_clip_count <= max_clipped_tokens:
                    keep = keep_bool.to(M.dtype)
                else:
                    # Keep only top-K most violating negative-advantage tokens clipped.
                    candidate_scores = xD.masked_fill(~neg_to_clip, float("-inf")).view(-1)
                    k = min(max_clipped_tokens, neg_to_clip_count)
                    _, indices = torch.topk(candidate_scores, k, largest=True)

                    limited_clip_mask = torch.zeros_like(candidate_scores, dtype=torch.bool)
                    limited_clip_mask[indices] = True
                    limited_clip_mask = limited_clip_mask.view_as(xD)

                    final_keep = keep_bool.clone()
                    final_keep[neg_to_clip] = True
                    final_keep[limited_clip_mask] = False
                    keep = final_keep.to(M.dtype)

        M_clipped = M * keep
        exps.batch["response_mask"] = M_clipped

        total_tokens = M.to(torch.float32).sum().clamp_min(1.0)
        frac_clipped = 1.0 - (M_clipped.to(torch.float32).sum() / total_tokens).item()

        pos_mask = (A > 0).to(M.dtype)
        neg_mask = (A < 0).to(M.dtype)
        total_pos = (M * pos_mask).to(torch.float32).sum().clamp_min(1.0)
        total_neg = (M * neg_mask).to(torch.float32).sum().clamp_min(1.0)
        frac_clipped_pos = 1.0 - ((M_clipped * pos_mask).to(torch.float32).sum() / total_pos).item()
        frac_clipped_neg = 1.0 - ((M_clipped * neg_mask).to(torch.float32).sum() / total_neg).item()

        metrics = {
            "frac_clipped": frac_clipped,
            "frac_clipped_pos": frac_clipped_pos,
            "frac_clipped_neg": frac_clipped_neg,
            "stdD": std_d.item(),
        }
        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "epsilon": 1e-6,
            "mu": 8.5,
            "max_frac": 1e-4,
        }
