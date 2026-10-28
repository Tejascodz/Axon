import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nyaya import scoring  # noqa: E402


def test_softmin_limits():
    S = torch.tensor([[[0.9, 0.2, 0.8]]])
    assert torch.allclose(scoring.softmin(S, tau=1e-3), torch.tensor([[0.2]]), atol=1e-2)
    assert torch.allclose(scoring.softmin(S, tau=1e3), S.mean(-1), atol=1e-3)


def test_mask_ignores_padding():
    S = torch.tensor([[[0.9, 0.8, -5.0]]])
    mask = torch.tensor([[True, True, False]])
    assert scoring.softmin(S, mask, tau=1e-3).item() > 0.79
    assert abs(scoring.chamfer(S, mask).item() - 0.85) < 1e-5


def test_gate_zero_cannot_veto():
    Q = F.normalize(torch.randn(1, 3, 8), dim=-1)
    D = F.normalize(torch.randn(4, 5, 8), dim=-1)
    gates = torch.tensor([[1.0, 1.0, 0.0]])
    full = scoring.score(Q, D, "softmin", tau=1e-3, gates=gates)
    two = scoring.score(Q[:, :2], D, "softmin", tau=1e-3)
    assert torch.allclose(full, two, atol=1e-2)


def test_and_beats_sum_on_one_missing_clause():
    # relevant doc: all 3 clauses matched moderately; negative: two perfect, one missing
    S = torch.tensor([[[0.7, 0.7, 0.7], [1.0, 1.0, 0.3]]])
    assert scoring.chamfer(S)[0, 0] < scoring.chamfer(S)[0, 1]       # sum ranks the wrong doc first
    assert scoring.softmin(S, tau=0.05)[0, 0] > scoring.softmin(S, tau=0.05)[0, 1]


def test_chunked_matches_full():
    Q = F.normalize(torch.randn(2, 4, 8), dim=-1)
    D = F.normalize(torch.randn(50, 6, 8), dim=-1)
    a = scoring.score(Q, D, "softmin")
    b = scoring.score_chunked(Q, D, chunk=7, kind="softmin")
    assert torch.allclose(a, b)


def test_grouped_mean_and_matches_flat_chamfer():
    # one clause holding every token == plain ColBERT
    Q = F.normalize(torch.randn(2, 1, 5, 8), dim=-1)
    D = F.normalize(torch.randn(9, 6, 8), dim=-1)
    a = scoring.score_grouped(Q, D, "chamfer")
    b = scoring.score(Q[:, 0], D, "chamfer")
    assert torch.allclose(a, b, atol=1e-6)


def test_grouped_and_vetoes_missing_clause():
    # doc 0 matches both clauses partly; doc 1 matches clause A perfectly and clause B not at all
    e = torch.eye(4)
    Q = torch.stack([e[0], e[1]])[None, :, None, :]                   # [1, 2 clauses, 1 token, 4]
    D = torch.stack([F.normalize(e[0] + e[1], dim=0)[None].repeat(2, 1),
                     torch.stack([e[0], e[2]])])                       # [2 docs, 2 tokens, 4]
    s = scoring.score_grouped(Q, D, "softmin", tau=0.05)
    assert s[0, 0] > s[0, 1]


def test_splitter_numbered_lists_and_prose():
    from nyaya.model import split_clauses
    q = "Find a book that meets these criteria: 1.Covers post-Cold War conflicts, 2.Includes expert essays, 3.Questions interventions."
    assert split_clauses(q) == ["Covers post-Cold War conflicts", "Includes expert essays", "Questions interventions"]
    assert split_clauses("Find a person: 1) born in 1950s, 2) won 3 awards") == ["born in 1950s", "won 3 awards"]
    assert split_clauses("What is the 2. best way to sell 1.5 million shares?") == ["What is the 2. best way to sell 1.5 million shares?"]
    assert len(split_clauses("A reporter who covered wars. The book has essays. It asks hard questions.", mode="sentences")) == 3
    assert len(split_clauses("A reporter who covered wars. The book has essays.")) == 1      # prose untouched by default
