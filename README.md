# Sleep Framework Continual Learning Lab 🧠💤

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)]()
[![Paper](https://img.shields.io/badge/arXiv-2606.03979-b31b1b.svg)](https://arxiv.org/html/2606.03979)
[![Google Research](https://img.shields.io/badge/Google%20Research-Blog%20Post-4285F4.svg)](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational, transparent PyTorch laboratory and deep-dive guide exploring the principles behind Google's **Sleep Framework** for continual learning in language models.

---

> [!IMPORTANT]
> **Independent Educational Implementation & Academic Disclaimer**
> 
> This repository is an independent educational laboratory designed to illuminate the underlying mathematical and architectural mechanics of continual learning, multi-timescale memory, and wake-sleep consolidation.
> 
> **This is NOT Google's official Sleep implementation, Hope architecture, or released model weights.** No proprietary code or weights were used. It is a synthetic PyTorch reproduction of the core mechanics designed to be fast, clear, reproducible on a single CPU core, and completely transparent about what works and why.

---

## 📑 Table of Contents

- [The Core Problem: Catastrophic Forgetting](#the-core-problem-catastrophic-forgetting)
- [How the Sleep Framework Works](#how-the-sleep-framework-works)
  - [The Continuum Memory Hierarchy](#the-continuum-memory-hierarchy)
  - [The 4-Phase Wake-Sleep Lifecycle](#the-4-phase-wake-sleep-lifecycle)
- [Laboratory Architecture & Benchmark](#laboratory-architecture--benchmark)
- [Experimental Results](#experimental-results)
  - [Analysis of Findings](#analysis-of-findings)
- [Repository Structure](#repository-structure)
- [Quickstart Guide](#quickstart-guide)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Unit Tests](#running-unit-tests)
  - [Running the Continual Learning Benchmark](#running-the-continual-learning-benchmark)
- [Deep Dive: Code Walkthrough](#deep-dive-code-walkthrough)
  - [1. Parameter Expansion (`FrozenExpandable`)](#1-parameter-expansion-frozenexpandable)
  - [2. Upward Distillation (`soft_distill`)](#2-upward-distillation-soft_distill)
  - [3. Synthetic Dreaming & Rehearsal (`select_dreams`)](#3-synthetic-dreaming--rehearsal-select_dreams)
  - [4. Imitation Reward (`concepts.py`)](#4-imitation-reward-conceptspy)
- [In-Depth Theoretical Companion (`ARTICLE.md`)](#in-depth-theoretical-companion-articlemd)
- [Scientific Honesty: Lab vs. Full Paper](#scientific-honesty-lab-vs-full-paper)
- [References & Citation](#references--citation)

---

## The Core Problem: Catastrophic Forgetting

When you prompt a modern Large Language Model (LLM) with a 50-page technical manual, it can answer queries about the document accurately while it sits inside the active attention context window. But as soon as that session terminates, the context is cleared—the model retains **zero** persistent memory of the document.

The intuitive remedy is to fine-tune the model on the new document. However, doing so triggers **catastrophic forgetting**:

```
Task A (Historical Knowledge) ──────────► [Train Model] ──────────► Task A: 99.8% Accuracy
                                                │
                                    Adapt to Task B (New Data)
                                                │
                                                ▼
                                         [Updated Model]
                                         Task B: 99.4% Accuracy  (Learned new task!)
                                         Task A: 52.9% Accuracy  (Catastrophic collapse to random guess!)
```

Because deep neural networks rely on distributed overlapping weight representations, backpropagating gradients for Task B overhauls parameter pathways that were critical for Task A.

---

## How the Sleep Framework Works

The paper *"Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories"* (Behrouz et al., 2026) proposes separating online interaction from persistent memory consolidation:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 WAKE PHASE                                      │
│  • Online user interactions & document intake                                   │
│  • Fast, ephemeral adaptation via short-term local parameters (Teacher)        │
│  • Core base model remains stable and protected                                 │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 SLEEP PHASE                                     │
│  1. Expand Capacity: Allocate fresh low-rank parameter capacity (Residual/LoRA) │
│  2. Upward Distillation: Transfer wake insights from Teacher into Student       │
│  3. Synthetic Dreaming: Generate synthetic prompts to practice edge cases       │
│  4. Rehearsal: Consolidate durable memories without overwriting historical data │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The Continuum Memory Hierarchy

The Sleep paradigm distinguishes three distinct tiers of memory:
1. **Contextual Memory**: Ephemeral activations in the KV-cache during inference (bounded by context window size).
2. **External Memory**: Search indices, vector databases, and retrieval stores (accurate, inspectable, but requires retrieval routing).
3. **Parametric Memory**: Model weights and low-rank residuals. Parametric memory enables generalized, retrieval-free reasoning, but requires careful consolidation to avoid destructive interference.

### The 4-Phase Wake-Sleep Lifecycle

```
           [Frozen Base Model (Task A)]
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
    (Wake Adaptation)   (Capacity Expansion)
   [Temporary Teacher]   [Frozen Base + Rank-8 Adapter]
             │                   │
             └─────────┬─────────┘
                       │
             [Upward Distillation]
             (Soft KL divergence loss)
                       │
                       ▼
             [Synthetic Dreaming]
     (Filter high disagreement + high confidence)
                       │
                       ▼
             [Dream Rehearsal]
             (Final Consolidated Model)
```

1. **Wake Adaptation**: When new data (Task B) arrives, an unconstrained temporary teacher model rapidly adapts to the new domain.
2. **Capacity Expansion**: Rather than modifying the base model's existing weights, the architecture freezes the historical weights and dynamically allocates new low-rank adapter capacity.
3. **Upward Knowledge Distillation**: The student model learns from the temporary teacher's soft output probability distributions across synthetic unlabeled inputs using temperature-scaled Kullback–Leibler (KL) divergence.
4. **Synthetic Dreaming**: The student and teacher evaluate candidate synthetic problems. Samples exhibiting high model disagreement paired with confident teacher labels are selected as "dreams" and rehearsed.

---

## Laboratory Architecture & Benchmark

To make these mechanics crystal clear, this repository implements a clean, deterministic PyTorch continual learning laboratory:

- **Input Space**: 2D Gaussian coordinates + 1 task indicator feature: $\mathbf{x} = [x_1, x_2, \text{task\_id}] \in \mathbb{R}^3$.
- **Task A**: Predict whether the first coordinate is positive ($x_1 > 0$).
- **Task B**: Predict whether the second coordinate is positive ($x_2 > 0$).
- **Model Architecture**:
  - `SmallNet`: 3-input $\to$ 24-neuron hidden layer ($\tanh$) $\to$ 2-class linear output head.
  - `FrozenExpandable`: Frozen base copy of `SmallNet` with an added low-rank bottleneck residual adapter:
    $$\mathbf{h} = \tanh(\mathbf{W}_{\text{base}} \mathbf{x} + \mathbf{b}_{\text{base}})$$
    $$\hat{\mathbf{y}} = \mathbf{W}_{\text{head}} \mathbf{h} + \mathbb{I}_{\text{task}=B} \cdot \left( \mathbf{W}_{\text{up}} \mathbf{W}_{\text{down}} \mathbf{h} \right)$$
    where $\mathbf{W}_{\text{down}} \in \mathbb{R}^{8 \times 24}$ and $\mathbf{W}_{\text{up}} \in \mathbb{R}^{2 \times 8}$.

---

## Experimental Results

The benchmark evaluates five training stages across **3 independent random seeds** on held-out test sets ($N=1,400$ per task). Results are saved in [`results/benchmark.json`](results/benchmark.json):

| Stage | Task A (Old) Accuracy | Task B (New) Accuracy | Note |
| :--- | :---: | :---: | :--- |
| **1. Baseline (Old Checkpoint)** | **99.79%** $\pm$ 0.08% | 51.40% $\pm$ 1.09% | Trained solely on Task A; chance level on Task B. |
| **2. Naive Sequential Fine-Tuning** | 52.95% $\pm$ 1.42% | **99.38%** $\pm$ 0.39% | **Catastrophic Forgetting**: Task A collapses to near random chance (~50%). |
| **3. Experience Replay (96 buffer)** | 94.14% $\pm$ 3.77% | 99.12% $\pm$ 0.16% | Partial retention of Task A, but notable degradation and high variance. |
| **4. Expanded + Distillation (Sleep)** | **99.79%** $\pm$ 0.08% | **99.10%** $\pm$ 0.42% | **Perfect Task A retention**; Task B successfully mastered via teacher KL distillation. |
| **5. Expanded + Dream Rehearsal** | **99.79%** $\pm$ 0.08% | 98.79% $\pm$ 0.57% | Rehearsing filtered synthetic dreams maintains stability. |

### Analysis of Findings

1. **Catastrophic Forgetting is Real**: In naive sequential fine-tuning, Task A accuracy instantly drops by **46.84 percentage points** (from 99.79% to 52.95%).
2. **Rehearsal Buffers Are Imperfect**: Preserving 96 past exemplars mitigates forgetting (recovering to 94.14%), but incurs memory storage overhead and still suffers a 5.65% drop.
3. **Architectural Gating vs. Learned MoE Routing**: In `FrozenExpandable`, Task A logits are 100% mathematically invariant by design because historical weights are frozen and the adapter is modulated by task identity. In real LLMs, routing must be learned dynamically, which introduces routing errors.
4. **Distillation Efficiency**: Knowledge distillation transfers new task capabilities cleanly into the low-rank adapter without requiring access to original Task B ground-truth labels during the consolidation phase.

---

## Repository Structure

```text
sleep_framework_lab/
├── README.md               # You are here: comprehensive project documentation
├── ARTICLE.md              # 800+ line Medium-style deep dive & research critique
├── BENCHMARK_NOTES.md      # Detailed benchmark interpretation & methodology notes
├── requirements.txt        # Minimal dependencies (PyTorch)
├── sleep_lab.py            # Complete runnable benchmark & architecture implementation
├── concepts.py             # Conceptual helpers: Levenshtein distance & imitation reward
├── results/
│   └── benchmark.json      # Raw JSON output across multiple experimental seeds
└── tests/
    └── test_lab.py         # Unit tests covering distillation, routing, and selection
```

---

## Quickstart Guide

### Prerequisites

- Python 3.10, 3.11, or 3.12
- PyTorch 2.1+ (CPU or CUDA/MPS; runs lightning fast on standard CPU)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdityaInamdar334/sleep-framework-lab.git
   cd sleep-framework-lab
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # macOS / Linux
   # .venv\Scripts\activate       # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Running Unit Tests

The test suite validates dataset synthesis, logit invariance, distillation loss calculations, dream filtering, and end-to-end execution:

```bash
python3 -m unittest discover -s tests -v
```

Expected output:
```text
test_edit_distance (test_lab.ConceptTests.test_edit_distance) ... ok
test_imitation_reward (test_lab.ConceptTests.test_imitation_reward) ... ok
test_dream_selection (test_lab.LabTests.test_dream_selection) ... ok
test_full_quick_benchmark (test_lab.LabTests.test_full_quick_benchmark) ... ok
test_old_exact (test_lab.LabTests.test_old_exact) ... ok
test_soft_distill (test_lab.LabTests.test_soft_distill) ... ok
test_task_labels (test_lab.LabTests.test_task_labels) ... ok

----------------------------------------------------------------------
Ran 7 tests in ~0.6s

OK
```

### Running the Continual Learning Benchmark

#### 1. Ultra-Fast Smoke Test (< 1 second)
```bash
python3 sleep_lab.py --seeds 1 --quick --output results/smoke.json
```

#### 2. Full 3-Seed Statistical Benchmark (~2 seconds)
```bash
python3 sleep_lab.py --seeds 3 --output results/benchmark.json
```

The script will output the summary table in JSON format and write detailed run logs directly to `results/benchmark.json`.

---

## Deep Dive: Code Walkthrough

### 1. Parameter Expansion (`FrozenExpandable`)

In [`sleep_lab.py`](sleep_lab.py), the `FrozenExpandable` class wraps the base model, freezes all original weights, and attaches low-rank matrices $\mathbf{W}_{\text{down}}$ and $\mathbf{W}_{\text{up}}$:

```python
class FrozenExpandable(nn.Module):
    def __init__(self, base: SmallNet, rank: int = 8):
        super().__init__()
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad_(False)  # Lock historical parameters
        
        width = self.base.head.in_features
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, 2, bias=False)
        nn.init.normal_(self.down.weight, std=0.03)
        nn.init.zeros_(self.up.weight)  # Zero initialization guarantees identity mapping at step 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.base.features(x)
            old_logits = self.base.head(h)
        residual = self.up(self.down(h))
        is_new = x[:, 2:3]  # Modulate via task indicator
        return old_logits + is_new * residual
```

### 2. Upward Distillation (`soft_distill`)

Rather than training on hard binary labels $\{0, 1\}$, the student learns from the soft logit distributions produced by the teacher using temperature-scaled Kullback–Leibler divergence:

```python
def soft_distill(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                 temperature: float = 2.0) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean',
    ) * (temperature ** 2)
```

### 3. Synthetic Dreaming & Rehearsal (`select_dreams`)

During "sleep", the model generates candidate problems. Instead of rehearsing everything, it filters for candidates where:
1. The teacher is confident ($\ge 65\%$).
2. The student and teacher disagree most strongly ($|\mathbf{p}_{\text{student}} - \mathbf{p}_{\text{teacher}}|$):

```python
def select_dreams(student: FrozenExpandable, teacher: SmallNet,
                  candidates: torch.Tensor, top_n: int = 160,
                  random_n: int = 40, seed: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
    student.eval()
    teacher.eval()
    with torch.no_grad():
        tprob = teacher(candidates).softmax(-1)
        sprob = student(candidates).softmax(-1)
        confidence, pseudo_y = tprob.max(-1)
        scores = (sprob - tprob).abs().sum(-1)
        
        # Discard low-confidence teacher hallucinations
        scores = torch.where(confidence >= 0.65, scores, -torch.ones_like(scores))
        order = scores.argsort(descending=True)
        high = order[scores[order] >= 0][:top_n]
        ...
    return candidates[indices], pseudo_y[indices]
```

### 4. Imitation Reward (`concepts.py`)

In [`concepts.py`](concepts.py), we illustrate how the Sleep paper defines **Learning to Imitate (LTI)**: rewarding sequence generation through a mixture of semantic correctness and bounded edit similarity:

```python
def imitation_reward(candidate: str, reference: str,
                     semantic_match: bool, gamma: float = 0.5,
                     max_normalized_distance: float = 0.5) -> float:
    """Illustrative mixture: human semantic bit + capped Levenshtein similarity."""
    length = max(len(candidate), len(reference), 1)
    normalized = edit_distance(candidate, reference) / length
    absolute = (1 - normalized) if normalized <= max_normalized_distance else 0.0
    return gamma * float(semantic_match) + (1 - gamma) * absolute
```

---

## In-Depth Theoretical Companion (`ARTICLE.md`)

This repository includes a full-length, publication-ready research article: [`ARTICLE.md`](ARTICLE.md). It breaks down:

- **Section 1–4**: Mathematical foundations of catastrophic forgetting, memory hierarchies, and the Wake-Sleep duality.
- **Section 5–9**: Detailed mechanics of low-rank expansion, upward distillation vs. standard distillation, and reinforcement learning for synthetic data.
- **Section 10–11**: **Critical evaluation**: What the published empirical results prove, where confounding variables exist, and why routing in Mixture-of-Experts remains the hardest open problem.
- **Section 12–21**: Complete step-by-step code construction and blueprint for scaling toward LLM production deployments.

---

## Scientific Honesty: Lab vs. Full Paper

To maintain rigorous scientific standards, here is an explicit side-by-side comparison between this educational laboratory and the research paper:

| Feature | Paper (*Behrouz et al., 2026*) | This Educational Lab |
| :--- | :--- | :--- |
| **Model Family** | 10M-token LLMs (Hope / Transformer) | 2-Layer Feedforward MLP |
| **Task Domain** | Natural language sequence modeling & reasoning | 2D Synthetic coordinate classification |
| **Task Routing** | Learned dynamic Mixture-of-Experts (MoE) router | Explicit input flag ($x_3$) + hard architectural gating |
| **Old Task Preservation** | Learned routing (subject to routing errors) | Guaranteed $100\%$ by freezing + structural gating |
| **Dream Selection** | Full parameter-gradient Fisher information & RL rewards | Softmax probability disagreement proxy |
| **Execution Cost** | Multi-GPU cluster training hours | < 2 seconds on a single CPU core |

---

## References & Citation

1. **Behrouz, A., Hashemi, F., Javanmard, A., & Mirrokni, V.** (2026). *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*. arXiv preprint [arXiv:2606.03979](https://arxiv.org/html/2606.03979).
2. **Google Research Blog** (October 27, 2026). *The Sleep framework: A new way for LLMs to retain knowledge in continual learning*. [research.google/blog](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/).

### BibTeX
```bibtex
@article{behrouz2026sleep,
  title={Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories},
  author={Behrouz, Ali and Hashemi, Farnoosh and Javanmard, Adel and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2606.03979},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE). Educational and research use is encouraged!
