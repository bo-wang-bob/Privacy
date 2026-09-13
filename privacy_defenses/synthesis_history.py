"""Local history of near-original visits, with no fitted decay or mixing weight."""
from __future__ import annotations

import torch


def _rank_counts(values):
    """Twice the zero-based midrank plus one, exactly represented as integers."""
    values = values.detach().cpu().double()
    if values.ndim != 1 or not len(values) or not torch.isfinite(values).all():
        raise ValueError("History ranking requires a nonempty finite vector.")
    order = torch.argsort(values, stable=True)
    _, counts = torch.unique_consecutive(values[order], return_counts=True)
    ends = counts.cumsum(0)
    ranks = torch.repeat_interleave(2 * ends - counts, counts)
    result = torch.empty(len(values), dtype=torch.long)
    result[order] = ranks
    return result


def midranks(values):
    """Equal values receive equal ranks; ranks are scaled to (0, 1)."""
    return _rank_counts(values).double() / (2 * len(values))


class ZeroRiskHistory:
    """Mean per-participation-round zero-risk frequency for each original ID.

    The current round is frozen. Only successful optimizer steps are observed;
    repeated local epochs are averaged within each original record and round.
    Missing history is zero. Bootstrap r=0 visits count as actual exposures.
    """

    def __init__(self):
        self.clients = {}

    def begin(self, client, round_index, size):
        state = self.clients.get(client)
        if state is None:
            state = dict(round=round_index, total=torch.zeros(size, dtype=torch.float64),
                         rounds=torch.zeros(size, dtype=torch.long),
                         zeros=torch.zeros(size, dtype=torch.long),
                         visits=torch.zeros(size, dtype=torch.long))
            self.clients[client] = state
        if size != len(state["total"]) or round_index < state["round"]:
            raise ValueError("Exposure history requires stable original IDs and increasing rounds.")
        if round_index > state["round"]:
            seen = state["visits"] > 0
            state["total"][seen] += state["zeros"][seen].double() / state["visits"][seen]
            state["rounds"][seen] += 1
            state["zeros"].zero_()
            state["visits"].zero_()
            state["round"] = round_index
        return state

    def values(self, client, round_index, size, indices):
        state = self.begin(client, round_index, size)
        indices = indices.detach().cpu().long()
        values = state["total"] / state["rounds"].clamp_min(1)
        return values[indices].clone(), state["rounds"][indices].clone()

    def observe(self, client, round_index, indices, used_risk):
        state = self.clients[client]
        indices = indices.detach().cpu().long()
        used = used_risk.detach().cpu().double()
        if (round_index != state["round"] or indices.ndim != 1 or used.shape != indices.shape
                or not torch.isfinite(used).all() or (used < 0).any() or (used > 1).any()
                or (indices < 0).any() or (indices >= len(state["total"])).any()):
            raise ValueError("Exposure observations must match the active round and original IDs.")
        state["zeros"].scatter_add_(0, indices, (used == 0).long())
        state["visits"].scatter_add_(0, indices, torch.ones_like(indices))


def history_assignment(raw_scores, history, original_risks):
    """Symmetric rank sum; preserve the original batch risk-weight multiset.

    Raw loss gap breaks joint-score ties, followed by stable batch order. This
    preserves the original ordering whenever all historical values are equal.
    The equal contribution of both ranks is a fixed design choice, not a tuned
    lambda hidden in configuration.
    """
    scores = raw_scores.detach().cpu().double()
    risks = original_risks.detach().cpu().clone()
    if scores.shape != history.shape or scores.shape != risks.shape:
        raise ValueError("Loss gaps, history and risks must align with original records.")
    # Sum integer rank counts before division, so equal sums remain exact ties
    # even for short batches whose length is not a power of two.
    joint_counts = _rank_counts(scores) + _rank_counts(history)
    joint = joint_counts.double() / (2 * len(scores))
    order = torch.argsort(scores, stable=True)
    order = order[torch.argsort(joint_counts[order], stable=True)]
    assigned = torch.empty_like(risks)
    assigned[order] = risks.sort().values
    return assigned, joint
