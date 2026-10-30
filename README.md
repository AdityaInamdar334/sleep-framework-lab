# Continual Learning Lab: Memory Consolidation, Distillation, and Forgetting

**A reproducible PyTorch study of catastrophic forgetting, experience replay, low-rank model expansion, and synthetic rehearsal, inspired by Google's *Sleep* framework.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Research-arXiv%3A2606.03979-B31B1B)](https://arxiv.org/abs/2606.03979)
[![License](https://img.shields.io/badge/License-MIT-2EA44F)](LICENSE)

A model can master a new task and still lose much of what it previously knew. This repository makes that failure observable in a small, controlled experiment and examines three ways to respond: **replay older examples**, **expand model capacity**, and **transfer behavior through distillation**.

The project is intentionally modest in scale. It runs on synthetic two-task classification data with a compact neural network, provides per-seed benchmark artifacts, and exposes the exact assumptions behind each result. It is designed for readers who want to **inspect the code, reproduce the measurements, and understand the limitations** before exploring continual learning in larger models.

> [!IMPORTANT]
> **Independent educational research project.** This is **not** Google's official Sleep framework, the Hope architecture, an LLM implementation, or a reproduction of the paper's reported benchmarks. It implements selected *conceptual analogues* using a synthetic classifier. No Google model weights or proprietary implementation are used.

## Key findings

Across three random seeds, the experiment observed:

- **Forgetting:** Old-task accuracy declined from **99.79% to 52.95%** after naive fine-tuning on a new task.
- **Replay:** Retaining **96 examples** from the original task improved old-task accuracy to **94.14%** after adaptation.
- **Expansion and distillation:** A frozen base with a task-gated, rank-8 residual adapter maintained **99.79%** old-task accuracy while reaching **99.10%** on the new task.
- **Synthetic rehearsal:** An additional rehearsal stage **did not improve** the new-task score in this experiment; it decreased from **99.10% to 98.79%**.

**Critical qualification:** The expanded model is given the correct task identity as an input feature. Its adapter is explicitly disabled for the old task, so old-task predictions remain unchanged **by construction**, not because the system has learned reliable task identification or continual memory consolidation.

## Contents

- [Research context](#research-context)
- [Experimental design](#experimental-design)
- [Results](#results)
- [Getting started](#getting-started)
- [How the implementation works](#how-the-implementation-works)
- [Reproducibility and validation](#reproducibility-and-validation)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Possible extensions](#possible-extensions)
- [Further reading and citation](#further-reading-and-citation)
- [License](#license)

## Research context

In sequential learning, training a neural network on a new distribution can interfere with parameters that supported earlier tasks. This is the **stability–plasticity problem**: a model needs enough flexibility to learn something new and enough stability to retain what it learned before.

The research paper [*Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories*](https://arxiv.org/abs/2606.03979), by Behrouz and collaborators, explores a wake/sleep learning paradigm with more sophisticated mechanisms than those demonstrated here. Its proposed *sleep* phase includes **Knowledge Seeding**, which transfers information into expanded model capacity, and **Dreaming**, which generates and evaluates synthetic learning experiences.

This repository asks a narrower question:

**What can we learn about forgetting, replay, capacity isolation, and teacher–student transfer using a small experiment where every training step is inspectable?**

The code studies analogous ideas. It does **not** implement the paper's multi-timescale Hope memory system, learned Mixture-of-Experts routing, on-policy sequence distillation, or reinforcement-learning-based Dreaming.

## Experimental design

### Tasks and network

Each input contains two Gaussian coordinates and an explicit binary task indicator:

```text
input = [x1, x2, task_id]

Task A (task_id = 0): classify whether x1 > 0
Task B (task_id = 1): classify whether x2 > 0
```

The baseline `SmallNet` is a small multilayer perceptron with three input features, one 24-unit `tanh` hidden layer, and a two-class output head.

The experiment first trains on Task A, then compares different approaches to Task B adaptation using independent held-out test sets.

### Compared approaches

**1. Naive sequential fine-tuning.** Continue updating the original model on Task B without rehearsal or constraints. This provides the forgetting baseline.

**2. Experience replay.** Mix new-task training batches with samples from a buffer of 96 Task A examples. This introduces historical supervision during adaptation.

**3. Low-rank expansion with distillation.** Freeze a copy of the Task A model. Add a rank-8 residual pathway for Task B and train it to approximate a Task B teacher's soft predictions using temperature-scaled KL divergence.

**4. Synthetic rehearsal.** Generate additional synthetic Task B inputs, prioritize confident teacher predictions where teacher and student disagree, and fine-tune the adapter on selected pseudo-labels.

The training flow is:

```text
                     Train on Task A
                            |
                   Task A checkpoint
                            |
             +--------------+---------------+
             |              |               |
             v              v               v
       Fine-tune on B   Replay A + B    Freeze base model
             |              |               |
             |              |        Train temporary B teacher
             |              |               |
             |              |        Add rank-8 B-only adapter
             |              |               |
             |              |        Distill teacher predictions
             |              |               |
             |              |        Optional synthetic rehearsal
             |              |               |
             +--------------+---------------+
                            |
              Evaluate on held-out A and B
```

### Why task identity matters

`FrozenExpandable` routes with the provided `task_id`. For Task A, the adapter's contribution is multiplied by zero. Therefore, its Task A outputs are exactly those of the frozen base, even after the adapter has been trained.

That is a useful example of **structural isolation**, but it removes the task-identification challenge that a realistic continually learning agent or learned expert router must solve. The strong retention result must be interpreted with this constraint in mind.

## Results

The following values are from the included [benchmark artifact](results/benchmark.json), averaged across **three independent random seeds (0, 1, and 2)**. Each value is a held-out accuracy percentage, reported as **mean ± sample standard deviation**.

| Training condition | Task A: old task | Task B: new task |
|---|---:|---:|
| Task A checkpoint, before adaptation | **99.79 ± 0.08%** | 51.40 ± 1.09% |
| Naive sequential fine-tuning | 52.95 ± 1.42% | **99.38 ± 0.39%** |
| Experience replay, 96 stored examples | 94.14 ± 3.77% | 99.12 ± 0.16% |
| Frozen base + rank-8 adapter + distillation | **99.79 ± 0.08%** | 99.10 ± 0.42% |
| Above, followed by synthetic rehearsal | **99.79 ± 0.08%** | 98.79 ± 0.57% |

### Interpreting the findings

**Sequential fine-tuning causes substantial interference in this setup.** The new-task model performs well on Task B while old-task accuracy falls by **46.84 percentage points**. This demonstrates catastrophic forgetting in the particular small classification experiment; it is not a measurement of forgetting in a language model.

**Replay provides a strong baseline.** A small buffer recovers much of the original-task performance, although its results vary across seeds. The buffer has a storage and data-retention cost that more advanced methods may try to reduce.

**The expanded adapter avoids old-task interference by design.** Its strong Task A accuracy is an expected consequence of freezing and gating, not evidence that it independently learned an effective MoE router. Distillation transfers useful Task B behavior to the adapter without using Task B ground-truth labels during that transfer stage.

**The dream-inspired stage is a negative result.** The additional pseudo-labeled rehearsal slightly **reduced** Task B accuracy in these runs. More synthetic data does not automatically mean better learning. This project's selection heuristic is substantially simpler than the paper's Dreaming approach, so the result neither verifies nor refutes the paper's findings.

Three seeds are useful for checking basic variability, but are insufficient for strong statistical or generalization claims. See [BENCHMARK_NOTES.md](BENCHMARK_NOTES.md) and the raw [results/benchmark.json](results/benchmark.json) for run-level detail.

## Getting started

### Requirements

- Python **3.10 or newer**
- PyTorch **2.1 or newer** (see [`requirements.txt`](requirements.txt))
- A CPU is sufficient for the included experiment; a GPU is not required

### Installation

Clone the repository:

```bash
git clone https://github.com/AdityaInamdar334/sleep-framework-lab.git
cd sleep-framework-lab
```

Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the tests

```bash
python -m unittest discover -s tests -v
```

The included test suite covers task-label construction, exact old-task logit preservation, distillation-loss behavior, synthetic-example selection, edit-distance helpers, and an end-to-end quick run. **Seven tests passed** in the checked project state.

### Run a quick experiment

```bash
python sleep_lab.py --seeds 1 --quick --output results/smoke.json
```

### Reproduce the reported benchmark

```bash
python sleep_lab.py --seeds 3 --output results/benchmark.json
```

The script prints the summary and writes a JSON file containing per-seed results, mean accuracies, and sample standard deviations. Runtime will depend on the machine and PyTorch installation; the experiment was designed to be lightweight on a CPU.

> **Tip:** The benchmark command writes to `results/benchmark.json`. Use another `--output` path if you want to preserve the committed reference results unchanged.

## How the implementation works

### `SmallNet`: a deliberately minimal baseline

The baseline is small enough to train repeatedly, allowing the experiment to isolate changes in training strategy rather than differences in model scale or pretraining data.

### `FrozenExpandable`: isolate new-task capacity

The expanded model keeps the original network frozen and learns a new residual transformation using two linear layers:

```python
class FrozenExpandable(nn.Module):
    def __init__(self, base: SmallNet, rank: int = 8):
        super().__init__()
        self.base = copy.deepcopy(base)

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        width = self.base.head.in_features
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, 2, bias=False)

        nn.init.normal_(self.down.weight, std=0.03)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.base.features(x)
            original_logits = self.base.head(features)

        residual = self.up(self.down(features))
        task_b_gate = x[:, 2:3]
        return original_logits + task_b_gate * residual
```

For Task A, `task_b_gate` is zero, so the residual cannot alter the original logits. The `up` projection also starts at zero, meaning the expanded network initially matches the frozen base for both tasks.

This is a **task-gated low-rank residual**, not a learned mixture-of-experts system and not a full implementation of the paper's Knowledge Seeding algorithm.

### `soft_distill`: transfer teacher predictions

The student minimizes KL divergence between its output distribution and the temporary Task B teacher's distribution:

```python
def soft_distill(student_logits, teacher_logits, temperature=2.0):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
```

Temperature scaling softens the distributions during training. The experiment uses generated Task B inputs for this transfer, with the teacher supplying probability targets. This is conventional classifier distillation, **not** on-policy autoregressive sequence distillation.

### `select_dreams`: select informative synthetic examples

The rehearsal stage samples synthetic Task B inputs and scores them using the **difference between teacher and student output probabilities**. It prioritizes high-disagreement candidates for which the teacher's maximum class probability is at least `0.65`, alongside a smaller random selection of other eligible candidates.

These scores are a heuristic. They are **not** the parameter-gradient, expert-perturbation, or reinforcement-learning reward mechanism described in the full Sleep paper.

The chosen examples receive teacher pseudo-labels and are used to fine-tune the task-specific adapter. As the benchmark shows, this additional step can reduce measured accuracy.

### `concepts.py`: a separate imitation-reward illustration

[`concepts.py`](concepts.py) implements edit-distance and an illustrative reward that mixes a binary semantic-match signal with normalized sequence similarity. The helpers show one way sequence-level imitation can be scored.

They are **not wired into the classifier's training loop** and should not be confused with the paper's complete Learning to Imitate objective.

## Reproducibility and validation

The included reference benchmark was captured with **PyTorch 2.10.0+cpu** using seeds `0`, `1`, and `2`.

- **Training data:** 1,400 examples per task, generated with seeded Gaussian random-number generators.
- **Evaluation data:** Separate, held-out sets of 1,400 examples per task.
- **Initial training:** 180 Adam steps on Task A.
- **Task B adaptation:** 220 Adam steps for the main sequential-training baselines.
- **Replay memory:** The first 96 Task A training examples.
- **Expanded model:** A frozen Task A base and a rank-8 Task B residual, with explicit task-ID gating.
- **Distillation:** Synthetic unlabeled Task B inputs and teacher-produced distributions.
- **Synthetic rehearsal:** Selected teacher-pseudo-labeled Task B inputs, using disagreement and confidence heuristics.
- **Reporting:** Mean and sample standard deviation over the three seeds, with full per-run values in JSON.

No held-out evaluation labels are used to train or select synthetic examples. Exact numbers can still vary with PyTorch version, platform, and execution details. The reference results describe this **specific controlled experiment**, not a standardized continual-learning benchmark suite.

## Project structure

```text
sleep-framework-lab/
├── ARTICLE.md             # Research-oriented explanatory companion
├── BENCHMARK_NOTES.md     # Experimental configuration and caveats
├── README.md              # Project overview and reproducibility guide
├── LICENSE                # MIT license
├── requirements.txt       # PyTorch dependency
├── sleep_lab.py           # Models, training methods, benchmark runner
├── concepts.py            # Edit distance and imitation reward examples
├── results/
│   └── benchmark.json     # Reference runs, means, and standard deviations
└── tests/
    └── test_lab.py        # Unit tests and quick integration test
```

For a longer discussion of the paper's research context and the differences between the paper and this experiment, see [ARTICLE.md](ARTICLE.md).

## Limitations

The repository intentionally leaves out several components necessary for a faithful reproduction of the research paper:

- **No language model or token generation.** The benchmark uses a two-class feedforward classifier, not an autoregressive LLM.
- **No Hope or multi-timescale memory architecture.** The example does not implement independently updated nested memory modules.
- **No learned task routing.** Task identity is directly encoded in the input, and the old-task preservation guarantee depends on that gate.
- **No full Knowledge Seeding.** The model demonstrates low-rank expansion and teacher–student distillation, but not the paper's pre-update memory transfer or generalized sequence-level distillation.
- **No paper-faithful Dreaming.** The experiment uses confidence-filtered pseudo-labeling and probability disagreement, not learned routing perturbations, parameter-gradient sample scoring, or RL-guided curriculum selection.
- **No claims about long-context reasoning or knowledge incorporation in LLMs.** The benchmark cannot measure those capabilities.

These limitations are central to interpreting the results. The project is best understood as a **teaching and prototyping environment for continual-learning mechanics**, not an implementation or validation of Google's research system.

## Possible extensions

For readers interested in making the experiment more realistic, useful next steps include:

1. **Remove the task-ID feature.** Evaluate how well a learned router can select adapters when task identity is unknown.
2. **Measure transfer and interference separately.** Add tasks with overlapping or conflicting decision boundaries instead of only independent coordinate-sign labels.
3. **Study memory efficiency.** Compare rehearsal-buffer size, number of trainable adapter parameters, and compute cost at matched accuracy targets.
4. **Improve synthetic selection.** Add selection ablations and compare disagreement-based sampling with random sampling, uncertainty sampling, and gradient-based criteria.
5. **Move to a language-model benchmark.** Introduce a small open-weight autoregressive model and evaluate sequential domain adaptation with fixed compute and independent retention tests.

Any larger-scale extension should clearly separate new experimental results from those reported in the original paper.

## Further reading and citation

**Research paper**

Ali Behrouz, Farnoosh Hashemi, Adel Javanmard, and Vahab Mirrokni. *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.* arXiv:2606.03979 (2026).

- [Read the paper](https://arxiv.org/abs/2606.03979)
- [Google Research's overview](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/)
- [Project's extended research notes](ARTICLE.md)

If you refer to the **Sleep framework**, please cite the original paper. If you refer to the **toy experiment or its measurements**, link to this repository and specify that it is an independent educational implementation.

```bibtex
@article{behrouz2026sleep,
  title={Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories},
  author={Behrouz, Ali and Hashemi, Farnoosh and Javanmard, Adel and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2606.03979},
  year={2026}
}
```

## License

Released under the [MIT License](LICENSE). This project is independent of Google Research and is not affiliated with or endorsed by the authors of the Sleep paper.
