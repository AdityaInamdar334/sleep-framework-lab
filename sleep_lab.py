"""Independent teaching lab for continual learning, not Google Sleep implementation.

Synthetic binary tasks:
  A: classify sign of x-coordinate
  B: classify sign of y-coordinate
Input includes an explicit task identity flag (third feature).

Produces honest numerical results for naive fine tuning, rehearsal, and a
frozen-base task-gated low-rank adapter trained with distillation and synthetic
teacher-labelled rehearsal. Structural routing preserves old-task logits by
construction; it is not learned MoE routing nor paper-faithful knowledge seeding.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def task_data(task: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if task not in (0, 1):
        raise ValueError("task must be 0 or 1")
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(n, 2, generator=g)
    flag = torch.full((n, 1), float(task))
    x = torch.cat((coords, flag), dim=-1)
    y = (coords[:, task] > 0).long()
    return x, y


class SmallNet(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(3, width), nn.Tanh())
        self.head = nn.Linear(width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class FrozenExpandable(nn.Module):
    """Frozen base with B-only low-rank residual. Not a routed MoE."""
    def __init__(self, base: SmallNet, rank: int = 8):
        super().__init__()
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad_(False)
        width = self.base.head.in_features
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, 2, bias=False)
        nn.init.normal_(self.down.weight, std=0.03)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.base.features(x)
            old_logits = self.base.head(h)
        residual = self.up(self.down(h))
        # Exact routing is provided by a known task ID, not learned inference.
        is_new = x[:, 2:3]
        return old_logits + is_new * residual


def optimize(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             steps: int, lr: float = 0.02, replay: tuple[torch.Tensor, torch.Tensor] | None = None,
             batch_size: int = 128, seed: int = 0) -> None:
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        idx = torch.randint(len(x), (batch_size,), generator=rng)
        xx, yy = x[idx], y[idx]
        if replay is not None:
            rx, ry = replay
            ri = torch.randint(len(rx), (batch_size // 4,), generator=rng)
            xx = torch.cat((xx, rx[ri]), dim=0)
            yy = torch.cat((yy, ry[ri]), dim=0)
        loss = F.cross_entropy(model(xx), yy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def soft_distill(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 temperature: float = 2.0) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean',
    ) * (temperature**2)


def distill(student: FrozenExpandable, teacher: SmallNet, x: torch.Tensor,
            steps: int, seed: int = 11, lr: float = 0.03) -> None:
    teacher.eval()
    student.train()
    rng = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([student.down.weight, student.up.weight], lr=lr)
    for _ in range(steps):
        idx = torch.randint(x.shape[0], (128,), generator=rng)
        xx = x[idx]
        with torch.no_grad():
            target = teacher(xx)
        loss = soft_distill(student(xx), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def select_dreams(student: FrozenExpandable, teacher: SmallNet,
                  candidates: torch.Tensor, top_n: int = 160,
                  random_n: int = 40, seed: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher pseudo-labels and a *logit-gradient proxy*, not param gradients.

    The candidate x distribution is synthetic and supplied by this benchmark.
    Real Sleep uses router perturbations, parameter-gradient scores and
    downstream-adaptation rewards, none of which are reproduced here.
    """
    student.eval()
    teacher.eval()
    with torch.no_grad():
        tprob = teacher(candidates).softmax(-1)
        sprob = student(candidates).softmax(-1)
        confidence, pseudo_y = tprob.max(-1)
        scores = (sprob - tprob).abs().sum(-1)
        # Avoid selecting teacher guesses of very low confidence.
        scores = torch.where(confidence >= 0.65, scores, -torch.ones_like(scores))
        order = scores.argsort(descending=True)
        high = order[scores[order] >= 0][:top_n]
        eligible = order[scores[order] >= 0]
        remaining = eligible[~torch.isin(eligible, high)]
        g = torch.Generator().manual_seed(seed)
        if len(remaining):
            remaining = remaining[torch.randperm(len(remaining), generator=g)[:random_n]]
        indices = torch.cat([high, remaining])
    return candidates[indices], pseudo_y[indices]


def rehearse_dreams(student: FrozenExpandable, dreams: tuple[torch.Tensor, torch.Tensor],
                     steps: int, seed: int = 30, lr: float = 0.01) -> None:
    x, y = dreams
    opt = torch.optim.Adam([student.down.weight, student.up.weight], lr=lr)
    rng = torch.Generator().manual_seed(seed)
    student.train()
    for _ in range(steps):
        idx = torch.randint(len(x), (min(128, len(x)),), generator=rng)
        loss = F.cross_entropy(student(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float((model(x).argmax(dim=-1) == y).float().mean().item() * 100)


@torch.no_grad()
def old_logits_unchanged(a: SmallNet, student: FrozenExpandable, x: torch.Tensor) -> bool:
    return bool(torch.equal(a(x), student(x)))


def run_once(seed: int = 0, quick: bool = False) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    start = time.perf_counter()
    n = 700 if quick else 1400
    a_train = task_data(0, n, 1000 + seed)
    b_train = task_data(1, n, 2000 + seed)
    a_test = task_data(0, 1400, 3000 + seed)
    b_test = task_data(1, 1400, 4000 + seed)
    old_memory = (a_train[0][:96], a_train[1][:96])
    base = SmallNet()
    optimize(base, *a_train, steps=80 if quick else 180, lr=0.02, seed=seed)

    def scores(m):
        return {"old": round(accuracy(m, *a_test), 2),
                "new": round(accuracy(m, *b_test), 2)}

    old = scores(base)
    naive = copy.deepcopy(base)
    optimize(naive, *b_train, steps=100 if quick else 220, lr=0.02, seed=seed + 30)
    naive_scores = scores(naive)
    replay = copy.deepcopy(base)
    optimize(replay, *b_train, steps=100 if quick else 220,
             lr=0.02, replay=old_memory, seed=seed + 30)
    replay_scores = scores(replay)

    # Wake: the temporary teacher has adapted to incoming task B.
    teacher = copy.deepcopy(base)
    optimize(teacher, *b_train, steps=100 if quick else 220, lr=0.02, seed=seed + 30)

    # Sleep: low-rank consolidation, with old base preserved by exact routing.
    student = FrozenExpandable(base, rank=8)
    b_synthetic, _ = task_data(1, 1700, seed + 9000)
    distill(student, teacher, b_synthetic, steps=100 if quick else 220, seed=seed + 55)
    consolidated = scores(student)
    candidates, _ = task_data(1, 1600, seed + 12000)
    dreams = select_dreams(student, teacher, candidates)
    rehearse_dreams(student, dreams, steps=35 if quick else 80, seed=seed + 50)
    dreamed = scores(student)
    assert old_logits_unchanged(base, student, a_test[0]), "old task must remain identical"
    return {
        "seed": seed,
        "old_checkpoint": old,
        "naive_finetune": naive_scores,
        "replay_96": replay_scores,
        "expanded_after_distillation": consolidated,
        "expanded_after_dream_rehearsal": dreamed,
        "selected_dreams": len(dreams[0]),
        "elapsed_sec": round(time.perf_counter() - start, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output", default="results/benchmark.json")
    args = p.parse_args()
    results = [run_once(seed, quick=args.quick) for seed in range(args.seeds)]
    stages = ["old_checkpoint", "naive_finetune", "replay_96",
              "expanded_after_distillation", "expanded_after_dream_rehearsal"]
    summary = {}
    for stage in stages:
        summary[stage] = {}
        for task in ("old", "new"):
            vals = [r[stage][task] for r in results]
            mean = sum(vals) / len(vals)
            stdev = math.sqrt(sum((v - mean)**2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0
            summary[stage][task] = {"mean": round(mean, 2), "std": round(stdev, 2)}
    output = {
        "description": "Synthetic two-task binary classification toy; not LLM or official Sleep reproduction",
        "task_identity_provided": True,
        "old_task_preservation": "guaranteed by explicit gating + frozen base",
        "seeds": args.seeds,
        "quick": args.quick,
        "summary": summary,
        "runs": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))
    print(f"Written: {path}")

if __name__ == "__main__":
    main()
