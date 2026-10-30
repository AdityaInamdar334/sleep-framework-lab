# Google Wants Language Models to Learn While They Sleep. We Built a PyTorch Lab to Understand Why.

## Inside the 2026 Sleep framework: continual learning, upward distillation, expandable memory, synthetic dreaming, and the limits of what the experiments establish

A language model reads a new research paper. It can answer questions about it while the paper is in its context window. Close that session, start another, and the model may behave as though the document never existed.

The obvious solution is to fine-tune the model on the paper.

But repeat that process over hundreds of unrelated documents and another problem emerges. The model may learn the new information while damaging capabilities it already had. Its parameters are an imperfect, shared storage medium. Writing something new can overwrite something old.

Now imagine a different lifecycle.

During the day, the model interacts with new information and accumulates temporary memories. At scheduled intervals, it stops serving new work. It opens additional capacity, transfers useful knowledge from temporary state into more durable parameters, and generates practice examples to improve itself. Then it returns to active use with an updated memory system.

This is the central proposal of **Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories**, a paper by Ali Behrouz, Farnoosh Hashemi, Adel Javanmard, and Vahab Mirrokni. Its revised arXiv version appeared in July 2026, and Google Research published an accessible explanation on October 27, 2026. [1][2]

The paper connects several existing research directions: continual learning, the Hope architecture, multi-timescale memory, low-rank parameter expansion, on-policy knowledge distillation, imitation learning, and reinforcement-learning-guided synthetic data generation.

The interesting claim is not that a model needs sleep in a biological sense. It is that **learning during active inference and consolidating knowledge into stable parameters may need to be separate computational operations**.

In this tutorial, we will first understand that argument at the level of model architectures and training objectives. Then we will build a compact PyTorch laboratory that reproduces three relevant mechanisms: sequential-learning interference, replay, and distillation into newly allocated low-rank capacity. Finally, we will use real results from our experiment to examine which conclusions are justified, and which still require much larger models and more rigorous tests.

We will not reproduce Google's reported 10-million-token experiments, long-context Hope architecture, or reinforcement-learning-based Dreaming system. The original paper and Google article do not link an official released training implementation as of October 30, 2026. The accompanying code is our independent educational experiment, not Google's code or a paper-equivalent reproduction.

There are no tables or LaTeX equations in this article. All code excerpts are plain Python and are included in the downloadable companion project.

## Table of Contents

1. Why a model that remembers a document can still forget it
2. The three kinds of memory we need to separate
3. What the Sleep paper actually proposes
4. Wake time and the continuum memory hierarchy
5. Why parameter expansion changes the forgetting problem
6. Knowledge Seeding: transfer before overwrite
7. On-policy distillation and why the sampling distribution matters
8. Learning to Imitate: retaining behavior as well as knowledge
9. Dreaming: synthetic data with a measurable purpose
10. What the paper's experiments really show
11. The results that deserve the most scrutiny
12. Build a simplified continual-learning laboratory
13. The baseline: train task A, then overwrite it with task B
14. Add a replay buffer
15. Add a frozen base and a low-rank expert
16. Consolidate knowledge through teacher-student distillation
17. Select and rehearse synthetic dreams
18. Our actual benchmark results and what they mean
19. How to design a more faithful replication
20. What would make this practical in deployed LLMs
21. Conclusion
22. References

---

## 1. Why a Model That Remembers a Document Can Still Forget It

Consider a technical support model deployed at a company. It knows the organization's hardware before a new router model arrives. A technician pastes the new manual into a conversation, and the model correctly explains the router's recovery procedure.

That answer demonstrates that the model can use context. It does not demonstrate that the model has updated its parameters.

In a conventional autoregressive Transformer, the conversation can alter the model's computations and its key-value cache, but it does not normally alter the pretrained weights. When the context disappears, the associated information may disappear with it. A system can retain a copy in an external database, but that is a separate memory mechanism.

Now train the model on the new manual instead. Some facts might become accessible without resupplying the source. Yet the update also changes shared weights that participate in many unrelated predictions.

The resulting failure is called *catastrophic forgetting*: progress on new training distributions can come with substantial regression on old ones.

This is not simply a consequence of poor prompt engineering. It follows from how distributed parameter representations overlap and from how gradient updates optimize the current training distribution rather than every task the model has ever learned.

The most basic continual-learning experiment captures this phenomenon with only two tasks. Train on A. Evaluate A. Train on B. Evaluate A again. If A collapses while B improves, the training process has acquired new capabilities at the expense of old ones.

The challenge is to make sequential adaptation less destructive while keeping the total cost, memory usage, and inference latency realistic.

## 2. The Three Kinds of Memory We Need to Separate

Discussions of LLM memory often mix three quite different ideas.

**Contextual memory** is the information available in the current input sequence, including prior dialogue, retrieved passages, tool results, and intermediate reasoning. It is flexible and often highly precise, but bounded by what can be processed, stored, or retrieved into the present context.

**External memory** lives outside the model weights. It may be a vector database, search index, SQL database, document store, or structured agent state. Its advantages include explicit updates, deletions, provenance, and relatively easy inspection. A model must still retrieve and use it correctly.

**Parametric memory** is carried by the model's learned weights and related persistent trainable components. It can support generalization and avoid repeated retrieval, but it is difficult to edit precisely, verify fact by fact, or erase reliably.

Google's Sleep proposal is primarily about **turning short-lived knowledge into longer-lasting parametric knowledge**. It is not merely a scheduler that summarizes yesterday's conversations into text files.

Those two forms of memory management may coexist in a practical system. A retrieval index can serve as authoritative, source-linked storage. Parameter consolidation can target patterns, procedures, habits, or frequently useful abstractions. One is not a universal replacement for the other.

There is also an important distinction between *remembering an answer* and *learning a general rule*. A system that recalls one exact router command has not necessarily acquired the concept of network recovery. A system that generalizes from new examples may still fail to recall the original command precisely.

A serious continual-learning evaluation needs to measure both factual retention and generalization, and should not treat improved benchmark accuracy as proof of perfect memorization.

## 3. What the Sleep Paper Actually Proposes

The framework organizes learning into two broad operating phases.

In the **wake phase**, the model actively receives new information. Its faster-changing memory components can adapt to the incoming stream. These representations are useful but potentially fragile because new input may displace them.

In the **sleep phase**, the model focuses its computation on previously encountered information. The paper divides this phase into **Memory Consolidation** and **Dreaming**.

Memory Consolidation provides new low-rank expert capacity in a slower-changing memory block. It then transfers behavior and knowledge from a faster or earlier state into that capacity using a teacher-student training process. After consolidation, temporary parameters can be reset for future use.

Dreaming generates synthetic examples, introduces exploration through expert routing, evaluates candidate training examples, and selects useful experiences for additional learning. The goal is not to replay every observed token indefinitely. It is to construct practice that reinforces or improves the model's future behavior.

The distinction matters: **consolidation attempts to preserve; dreaming attempts to improve**.

Both stages operate within a defined machine-learning architecture and optimization scheme. The neuroscience terminology is an analogy and a design inspiration. The experiments do not establish that LLMs undergo biological sleep, autonomously invent new optimization rules, or become generally self-aware. [1][2]

A useful mental model is:

```text
Incoming experience
        |
        v
WAKE
  fast adaptation
  context / temporary state
        |
        v
SLEEP
  1. open slower memory capacity
  2. distill teacher into expanded student
  3. reset disposable fast parameters
  4. generate and select practice examples
  5. refine using rewarded practice
        |
        v
NEXT WAKE CYCLE
  old knowledge should remain useful
  new knowledge should be accessible
```

This diagram captures the conceptual lifecycle. The actual paper integrates it with multi-level memory and model-specific training operations, not a single universal API that can be attached to every Transformer.

## 4. Wake Time and the Continuum Memory Hierarchy

The paper builds on *Nested Learning* and the Hope architecture, which treat a learner as a collection of components with different update frequencies. [3]

Think of a memory component updated every token, another updated every thousand tokens, and a third updated every ten thousand tokens. The faster one can respond quickly but may be unstable. The slower one changes less frequently and can retain patterns over longer intervals.

The crucial insight is that memory stability is relative. A parameter block can be durable compared with a token-level update and fragile compared with a still slower block.

This motivates more than one consolidation boundary.

A fast block may need to transfer its knowledge into an intermediate block repeatedly before the intermediate block transfers its own accumulated state to a slower block.

The sleep framework schedules these transfers around the update frequencies of the memory modules. Importantly, it aims to consolidate before an update overwrites the state being transferred.

This last point is subtle and essential.

Suppose a fast memory is about to apply an update that destroys part of its previous state. Distilling from the memory after that destructive update would be too late. The paper instead constructs a prospective update, preserves a teacher based on the pre-update state, and trains expanded student capacity to retain that teacher behavior before the state change becomes final.

For engineers, this resembles a transaction with a checkpoint and a staged commit. The memory update and the consolidation must be ordered so that the old state remains available while its useful information is transferred.

The paper's implementation discussion also notes that new experts do not necessarily require allocating entirely new tensor shapes at runtime. Capacity can be preallocated and masked until a sleep phase activates it.

That makes the idea more compatible with systems that prefer static memory layouts, although it does not eliminate the compute and storage costs of inactive or newly activated capacity.

## 5. Why Parameter Expansion Changes the Forgetting Problem

Traditional fine-tuning updates existing weights to absorb new training information. When the same parameters are responsible for older behavior, gradient interference becomes unavoidable in some regions of the loss landscape.

One response is to constrain the updates. Elastic Weight Consolidation, for example, penalizes changes to parameters judged important for old tasks. Another is to replay old examples alongside new ones.

Sleep explores a different option: make additional representational capacity available where the new knowledge is being stored.

The paper models memory components as sparse Mixture-of-Experts blocks. During consolidation, a new low-rank expert can be added to a slower-frequency block so the student receives fresh parameters for the transferred knowledge.

A low-rank expert is attractive because it can express a useful correction with far fewer trainable parameters than a full dense matrix. Instead of updating every element of a large projection, the system learns two smaller matrices whose product acts as a structured change.

This is related to the basic idea of LoRA, but a low-rank expansion is not automatically equivalent to ordinary LoRA fine-tuning. Its role in the paper is part of a larger memory-management protocol involving expert selection, consolidation schedules, frozen parameters, and temporary-memory reset.

There is also a tradeoff: isolation and added capacity can reduce destructive interference, but they do not guarantee good transfer.

If new experts never share information with older ones, the result can become a collection of disconnected task specialists. If a router chooses the wrong expert, even perfectly preserved knowledge may be inaccessible. If experts grow indefinitely, total storage and operational complexity increase.

The paper reports keeping active parameter counts controlled in its evaluated configuration, but this does not imply that total allocated capacity or training compute are free. [1]

An important research question is therefore not merely *can we preserve yesterday's task?* It is *can we preserve knowledge, generalize across tasks, and locate the right knowledge without an oracle telling us which component to activate?*

## 6. Knowledge Seeding: Transfer Before Overwrite

The paper calls its consolidation procedure **Knowledge Seeding**. A smaller or less expanded state acts as the teacher. The student has newly expanded capacity intended to preserve the teacher's useful behavior even after the faster memory is updated or reset. [1]

The ordering is the heart of the method.

First, the system computes the pending update that would change a fast memory block, but does not immediately discard the previous state.

Second, it constructs a teacher representing the current pre-update behavior and a student representing the prospective post-update architecture plus fresh low-rank capacity.

Third, it trains that new capacity to reproduce useful teacher behavior. The existing student parameters are frozen during the transfer, so learning is concentrated in the newly opened memory.

Only after consolidation does the system apply the planned update and reset temporary expert capacity where appropriate.

This is why a simple fine-tuning loop followed by a validation test is not an accurate implementation of the paper.

The pre-update teacher, prospective student, and commitment ordering all matter. They determine which information is actually available at the moment of transfer.

**A useful engineering analogy** is copying live state into a new storage tier before recycling the original buffer. The copy is not sufficient unless the destination is validated before the source is reused.

In a model, validation is harder than checking byte equality. What needs preserving is useful behavior across relevant inputs, not a literal duplicate of parameter values.

That is where distillation becomes important.

## 7. On-Policy Distillation and Why the Sampling Distribution Matters

A teacher-student system can be trained using a fixed collection of teacher-generated examples. But the student may produce different sequences after deployment, so it can encounter contexts that were rare or absent in the teacher's fixed training set.

Generalized Knowledge Distillation, or GKD, addresses this by letting the student generate some of its own trajectories and then receiving feedback from the teacher on those trajectories. [4]

The Sleep paper builds on that idea. It describes training on both teacher-provided and student-sampled outputs. The student receives distribution-level feedback from the teacher rather than learning only to imitate a fixed list of correct final answers.

Why does the sampling distribution matter?

Imagine a student generating a mathematical solution that takes an unusual but plausible wrong turn. A teacher-only training collection may never show the student how to recover from that exact prefix. An on-policy distillation process can expose the student to its own problematic prefixes and guide the next-token distribution there.

This can reduce the mismatch between the states encountered during training and inference.

The paper's objective contains a mixing coefficient that determines how much weight goes to teacher-sampled versus student-sampled trajectories. It uses distribution divergence to compare the two models and explicitly avoids backpropagating through the sampling operation itself.

You can understand the central calculation without mathematical notation:

```python
import torch.nn.functional as F

def soft_distill(student_logits, teacher_logits, temperature=2.0):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
```

This is real, runnable PyTorch code for a KL-based teacher-student loss. **It is not the full GKD objective**. Our laboratory uses it for a two-class classifier, without autoregressive rollouts or token-by-token teacher feedback on student-generated sequences.

In the real sequence-model setting, one must also manage attention masks, teacher-forced log probabilities, generated continuation lengths, sampling policies, tokenization, and the computational cost of the additional teacher pass.

An incomplete reimplementation that ignores these details may produce a valid tensor loss but fail to reproduce the method.

## 8. Learning to Imitate: Retaining Behavior as Well as Knowledge

The authors identify a problem that can survive ordinary distillation: the student may store some useful information yet fail to employ it in the same ways as the teacher.

The paper adds a **Learning to Imitate**, or LTI, component.

A teacher generates a continuation. The system chooses a prefix from that continuation, asks the student to finish it, and scores the response on two dimensions.

One dimension measures whether the output has a similar meaning. Another measures similarity of the sequences themselves through a normalized edit-distance signal.

The semantic component matters because two valid explanations can use different words. The edit-distance component provides a more literal signal about reproducing the teacher's generated trajectory.

These rewards serve different purposes. A purely literal imitation objective can punish harmless paraphrasing. A purely semantic objective may fail to preserve useful structural features of the teacher's reasoning or style.

The paper combines the reward with distillation in an RL-oriented training objective. That is a substantive addition, not a minor post-processing step.

Here is an intentionally small illustration of the absolute-similarity part:

```python
from concepts import imitation_reward

score = imitation_reward(
    candidate="reset the router",
    reference="reboot the router",
    semantic_match=True,
    gamma=0.5,
)
print(round(score, 3))
```

The helper in the companion lab implements Levenshtein distance and combines it with a semantic-match Boolean supplied by the caller. It does not run an LLM, train a reward model, produce token-level RL gradients, or implement the authors' complete LTI algorithm.

This separation is important for reproducibility. A reward function illustrated in Python is not evidence that the corresponding reinforcement-learning training method has been reproduced.

## 9. Dreaming: Synthetic Data With a Measurable Purpose

The Dreaming stage tries to do more than preserve what the model saw during wake time.

The paper draws on Self-Adapting Language Models, or SEAL, where a model proposes updates and receives feedback based on whether those updates improve downstream performance. [5]

In Sleep, synthetic examples are generated using information the model has encountered. The authors perturb expert selection during generation so the model may combine knowledge from different regions of its memory architecture.

Then comes selection. If every synthetic example were equally helpful, generation alone would be sufficient. But self-generated data can be repetitive, wrong, low-value, or actively damaging.

The paper therefore uses gradient-based importance estimates to filter candidate dreams while keeping some random samples for diversity.

It also describes a more expensive evaluation loop: train an isolated model instance on a candidate dream, observe whether performance improves on a target evaluation, and use that outcome as a reward signal for refining the dream-generation policy.

This reveals an often overlooked distinction:

**A plausible synthetic example is not necessarily a useful training example.**

A model may write an impressive-looking explanation that introduces no new capability. Another example may be awkward but expose exactly the weakness that subsequent training should address.

A more faithful Dreaming implementation needs a meaningful task distribution, a reward or adaptation benchmark, a mechanism for preventing contamination of held-out evaluation data, and enough compute to evaluate candidate updates.

Our small lab uses a deliberately weaker proxy: it asks a temporary teacher to pseudo-label synthetic samples, scores their disagreement with the student, and selects difficult examples with a small random sample for diversity. It does **not** implement random MoE expert routing, parameter-gradient selection, isolated inner-loop fine-tuning rewards, or reinforcement-learning optimization.

That difference is not cosmetic. It is why our final result can test the value of a simple rehearsal stage but cannot validate the paper's Dreaming algorithm.

## 10. What the Paper's Experiments Really Show

The original study evaluates several domains, not just one language-model leaderboard. These are the authors' reported results from arXiv version 2, not numbers produced by the companion experiment. [1]

### Continual text classification

The researchers investigate sequential class learning on CLINC, Banking77, and DBpedia-derived data, using Llama-family backbones and a Hope-based memory architecture. They compare against in-context baselines, Elastic Weight Consolidation, an external-learner approach, and multi-level Hope variants.

The central comparison evaluates whether newly learned classes remain accessible after subsequent data streams. The plotted results favor the Hope and consolidation approach in the reported configurations.

These experiments matter because class-incremental learning creates a genuine sequential-learning requirement. Still, the exact gains depend on data ordering, head configuration, backbone, update schedule, and architectural capacity.

### Long-context reasoning and retrieval

The authors study multi-key retrieval and question answering using RULER's multi-key needle-in-a-haystack task, LongHealth, and QASPER. They also evaluate on BABILong, including contexts extending to 10 million tokens.

The reported Hope-based method maintains strong BABILong performance at lengths where several comparison systems degrade. This is the source behind the widely repeated 10-million-token headline.

But it is important to say what that headline does **not** mean.

It does not mean that a standard Transformer can suddenly accept ten million tokens through a conventional attention matrix with no additional cost. It also does not establish perfect recall of every arbitrary fact in a ten-million-token document.

The result is tied to the memory-system design, training procedure, benchmark task family, and fine-tuned setting. The paper explicitly notes that small-model performance deteriorated substantially without fine-tuning.

BABILong itself is an evaluation of reasoning over facts distributed in long distractor-filled contexts. It should not be confused with universal long-document understanding. [6]

### Sequential novel-language learning

The researchers evaluate translation involving Manchu and Kalamang, comparing separate learning against learning both sequentially.

Their multi-stage Hope variants retain more of the first language's benefit after learning the second. This is a particularly direct test of forgetting because acquiring the second task should not destroy the first.

The reported behavior is consistent with the motivation for multi-frequency memory. It does not prove the approach will work equally well on every language pair or every form of distribution shift.

### Knowledge incorporation

The SQuAD-based experiments study whether new factual information can become available when the original passage is no longer in context.

The paper reports **31.9%** for the base model in both the single-passage and 200-passage settings. Its four-level Sleep variant reaches **48.9%** and **46.2%**, respectively. The compared SEAL configuration reaches **46.7%** and **43.2%**.

The difference is not simply between a model with access to the passage and one without access. The authors evaluate no-context factual answering after an adaptation procedure. Their 200-passage setting includes **974 associated evaluation questions**.

The ablations are especially informative. Removing Dreaming drops the reported results to **35.7%** for a single passage and **36.2%** for the continued-pretraining setting. That suggests the Dreaming contribution is important in that particular experimental design.

It does not prove that every unsupervised synthetic-data loop improves every model. It demonstrates a comparison inside the authors' selected setup.

### Mathematical reasoning

For Qwen3-8B, the paper reports average-at-16 scores of **79.2 on AIME 2024**, **69.0 on AIME 2025**, and **46.1 on HMMT 2025** for the full Sleep configuration.

The base-model scores are **73.8**, **68.1**, and **42.4**, respectively. The paper's GRPO comparison scores **76.4**, **68.1**, and **44.9**. Its on-policy self-distillation comparison scores **76.6**, **67.4**, and **45.1**.

These are improvements, but their magnitude varies considerably across the three benchmarks. Small differences should not be presented as guaranteed gains across independent model deployments or evaluation harnesses.

The paper also studies Qwen3-1.7B, where the Sleep configuration reports **53.2**, **40.2**, and **29.3** on the same benchmark sequence.

These numbers depend on the evaluation's average-at-16 definition. They should not be casually equated with greedy, single-sample accuracy.

### Few-shot abstract reasoning

A small ARC-style evaluation using Llama-3.2-1B reports **80% success** for Sleep, compared with **72.5%** for SEAL and **10%** for a test-time-training baseline.

The appendix describes **11 training tasks and eight held-out tasks** after filtering. That makes the result an interesting proof of concept, but the limited task count and dream-based evaluation procedure make it especially unsuitable as evidence of broad, general reasoning superiority.

## 11. The Results That Deserve the Most Scrutiny

There are at least five reasons to read the reported improvements carefully.

First, the evaluations do not all test the same capability. Class retention, ultra-long retrieval, no-context factual answering, and mathematics require different forms of learning and different evaluation machinery.

Second, the architecture changes alongside training. Adding experts and using a Hope-style memory hierarchy is not the same intervention as changing only a loss function. A fair ablation must isolate capacity growth, routing, distillation, and synthetic rehearsal.

Third, the cost of sleeping is real. The appendix says that, at the same number of training steps, its SFT baseline is about **four times more efficient** than the proposed method. It also reports that when targeting the same benchmark performance, SFT required about **4.3 times, 3.6 times, and 4.8 times** the wall-clock time in the three math evaluations. Those two comparisons answer different questions, and neither should be stripped of its benchmark-specific conditions.

Fourth, continual learning needs a strong post-update evaluation protocol. A system may show a high average score because it improved on one task while quietly regressing on a rare older capability. Forward transfer, backward transfer, forgetting, adaptation compute, and memory growth should all be tracked.

Fifth, we need better reproducibility. The paper provides algorithms, experiments, and implementation details, but the linked paper page and Google Research announcement do not expose an official standalone code repository or downloadable trained checkpoint for the full Sleep system as of this writing. This makes independent replication harder.

The careful conclusion is that the authors report compelling evidence for combining hierarchical memory, expansion, consolidation, and synthetic practice. The research does not establish a fully autonomous continual-learning product that can learn arbitrary new knowledge without limits, oversight, or regressions.

Now let us build something we can actually run.

---

## 12. Build a Simplified Continual-Learning Laboratory

Our reproduction is intentionally much smaller than the paper's system.

We will use a two-layer PyTorch classifier with a small hidden representation, not a Transformer. It will learn two simple tasks presented sequentially.

**Task A:** given two Gaussian coordinates, predict whether the first coordinate is positive.

**Task B:** given the same kind of coordinates, predict whether the second coordinate is positive.

The model receives a third input feature indicating which task it is solving. This task identifier is available in all methods. It is an important experimental convenience and a serious limitation, because real continual-learning systems may have to infer the task or select an appropriate memory expert on their own.

We will train and compare five checkpoints: the model after A, naive fine-tuning on B, B fine-tuning with a replay buffer of A examples, a frozen A model with an expanded low-rank adapter distilled from a temporary B teacher, and the same expanded model after additional synthetic rehearsal.

This lab lets us measure forgetting and the effect of capacity isolation with known ground-truth labels. It cannot reproduce language generation, sequence-level on-policy distillation, complex Hope memory routing, or 10-million-token context retention.

### Set up the environment

Download the companion project, then run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

To execute all three seeded benchmark runs:

```bash
python sleep_lab.py --seeds 3 --output results/benchmark.json
```

For a faster smoke test:

```bash
python sleep_lab.py --seeds 1 --quick \
  --output results/smoke.json
```

No pretrained model downloads, GPUs, API keys, or private datasets are required.

The code was tested with **PyTorch 2.10.0+cpu**. The full three-seed experiment and the unit tests were executed locally for this article.

## 13. The Baseline: Train Task A, Then Overwrite It With Task B

First, construct a reproducible dataset.

```python
import torch

def task_data(task: int, n: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    coords = torch.randn(n, 2, generator=generator)
    task_id = torch.full((n, 1), float(task))
    x = torch.cat([coords, task_id], dim=-1)
    y = (coords[:, task] > 0).long()
    return x, y
```

The label comes from the first coordinate for Task A and the second for Task B. The third input feature says which task is active.

This is deliberately simple. We are not evaluating complex reasoning. We want a controlled setting in which the effect of sequential training is easy to see.

Use a small network:

```python
import torch.nn as nn

class SmallNet(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(3, width),
            nn.Tanh(),
        )
        self.head = nn.Linear(width, 2)

    def forward(self, x):
        return self.head(self.features(x))
```

Train the network on A first. Use held-out A examples to measure the old-task accuracy. Then continue updating the same parameters on B alone.

In the provided benchmark, the model trains on 1,400 synthetic examples for each task, with 1,400 independently generated evaluation examples per task. The optimization code uses Adam, and the run fixes its random seeds for reproducibility.

The important comparison is between the checkpoint after A and the checkpoint after B.

In our three-seed experiment, the model's old-task accuracy dropped from **99.79%** to **52.95%**, while its new-task accuracy reached **99.38%**.

That is a clean example of catastrophic interference. The network can learn either decision boundary. Naive sequential training does not preserve both reliably under these settings.

Do not read the 52.95% figure as a general forgetting rate for LLMs. It is the outcome of our specific tiny network, Gaussian data, task order, and optimizer schedule.

## 14. Add a Replay Buffer

Before building any new architecture, we need a conventional baseline that is surprisingly hard to beat: rehearsal.

During A training, save a small subset of its examples. When training on B, mix some saved A examples into each mini-batch.

The model is then exposed to the new distribution without entirely losing contact with the old one.

The critical part of the implementation looks like this:

```python
if replay is not None:
    replay_x, replay_y = replay
    selected = torch.randint(
        len(replay_x),
        (batch_size // 4,),
        generator=rng,
    )
    x_batch = torch.cat(
        [x_batch, replay_x[selected]], dim=0
    )
    y_batch = torch.cat(
        [y_batch, replay_y[selected]], dim=0
    )
```

The benchmark uses only 96 stored A examples and samples a replay batch equal to one quarter of the new-task batch size.

This yields **94.14% old-task accuracy** and **99.12% new-task accuracy**, averaged over three runs.

Replay is not perfect here. But it substantially improves retention without increasing the model's parameters.

This baseline creates a meaningful comparison for any sleep-inspired system. If a new architecture uses more memory, more compute, or explicit task routing, it should be tested against competent replay approaches, not just against naive fine-tuning.

There is also a privacy and data-retention consideration. Keeping old training examples may be impossible in some deployments. A synthetic replay system or a consolidated parameter memory could become attractive precisely when raw training examples cannot be stored indefinitely.

## 15. Add a Frozen Base and a Low-Rank Expert

Now consider a different design. Rather than let B training modify the parameters that already solve A, freeze the A model.

Add a small residual expert with a narrow intermediate dimension. Route B examples through this expert while sending A examples through the preserved base prediction.

```python
import copy
import torch.nn as nn

class FrozenExpandable(nn.Module):
    def __init__(self, base, rank=8):
        super().__init__()
        self.base = copy.deepcopy(base)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        width = self.base.head.in_features
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, 2, bias=False)
        nn.init.normal_(self.down.weight, std=0.03)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        with torch.no_grad():
            features = self.base.features(x)
            old_logits = self.base.head(features)

        residual = self.up(self.down(features))
        is_task_b = x[:, 2:3]
        return old_logits + is_task_b * residual
```

There are two separate lessons here.

The first is about capacity. The new expert receives trainable parameters that are distinct from the base network.

The second is about routing. We do not learn a sophisticated router. We provide an **explicit task-ID feature** and use it to multiply the residual by zero on old-task inputs.

This means the base model's outputs on Task A are unchanged by construction. It is a controlled demonstration of how architectural isolation can prevent interference.

It is **not evidence that the Google framework has solved expert routing or universal catastrophic forgetting**. Those problems are intentionally simplified away here.

The toy adapter has rank eight and uses two linear projections. This makes the additional trainable state small and makes the distillation step inexpensive on a CPU.

## 16. Consolidate Knowledge Through Teacher-Student Distillation

To mimic the separation between temporary learning and more durable memory, train a temporary teacher on B starting from the A checkpoint.

This teacher can become good at the new task while suffering on the old task. We only use it to teach the new expert what B requires.

Next, freeze the expanded model's base and train the adapter to match the teacher's predictions over a fresh stream of synthetic Task B inputs.

```python
teacher.eval()
student.train()

for step in range(220):
    indices = torch.randint(
        len(synthetic_b), (128,), generator=rng
    )
    batch = synthetic_b[indices]

    with torch.no_grad():
        target_logits = teacher(batch)

    predicted_logits = student(batch)
    loss = soft_distill(
        predicted_logits,
        target_logits,
        temperature=2.0,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

This is a simplified teacher-student consolidation loop. It resembles the idea of transferring behavior into new capacity but **does not reproduce the paper's pre-update teacher and prospective-update student**, its multiple memory frequencies, or the on-policy mixture of autoregressive sequences.

It is useful precisely because we can isolate and measure one hypothesis: can a frozen model acquire a new task through a small trainable residual without changing its old predictions?

After distillation, the lab reports **99.79% old-task accuracy** and **99.10% new-task accuracy**.

The old accuracy is identical to the original A checkpoint because of the explicit gate and frozen parameters, a structural guarantee verified by a unit test.

The new accuracy shows that the adapter learned to implement the B decision boundary through distillation.

This does not establish that the adapter would retain hundreds of unknown future tasks, discover task boundaries by itself, or generalize to long-context LLM memory.

## 17. Select and Rehearse Synthetic Dreams

The companion experiment has one final stage inspired by Dreaming.

We generate additional synthetic B inputs from the same Gaussian source used in our task definition. The B teacher supplies pseudo-labels. We compare the teacher's output probabilities with the student's probabilities and select examples with high disagreement, provided the teacher is reasonably confident.

We also include some randomly selected examples to keep the rehearsal set from collapsing onto only one narrow region.

The core scoring logic is:

```python
with torch.no_grad():
    teacher_probs = teacher(candidates).softmax(-1)
    student_probs = student(candidates).softmax(-1)

    confidence, pseudo_labels = teacher_probs.max(-1)
    disagreement = (
        student_probs - teacher_probs
    ).abs().sum(-1)

    disagreement = torch.where(
        confidence >= 0.65,
        disagreement,
        -torch.ones_like(disagreement),
    )
```

We call the disagreement a **logit-gradient proxy**, because it resembles a magnitude of a simple output-level cross-entropy learning signal. We do **not** compute the full parameter gradient for every dream, as the paper's gradient-based data selection describes.

There is no random MoE expert activation, no inner-loop reward from an independently fine-tuned model, and no policy optimization over dream generation.

This is a small, interpretable ablation: after teacher distillation has already worked, does extra pseudo-labeled training improve the adapter?

In our run, the answer is **no**.

Before dream rehearsal, new-task accuracy was **99.10%**. After it, accuracy was **98.79%**, a small decrease. Old-task accuracy remained **99.79%**, again due to explicit routing.

This is not surprising in retrospect. The adapter was already strong; synthetic pseudo-labels can introduce noise and further optimization can slightly shift the classifier boundary.

The negative result is useful. It shows why the paper's more sophisticated dream selection and downstream improvement rewards are important. More synthetic data is not automatically more learning.

It would be misleading to present this toy result as a failure of Google's Dreaming method, just as it would be misleading to claim we reproduced that method successfully.

## 18. Our Actual Benchmark Results and What They Mean

The results below come from **three real local runs**, with random seeds 0, 1, and 2. They were generated using CPU PyTorch 2.10.0+cpu on a synthetic two-task classifier. Each number is the mean held-out accuracy; the accompanying variation is the sample standard deviation across those three seeds.

**After Task A training:** Old-task accuracy was **99.79% ± 0.08**. The model's Task B accuracy was **51.40% ± 1.09**, near chance before dedicated B training.

**After naive sequential fine-tuning:** Old-task accuracy dropped to **52.95% ± 1.42**, while new-task accuracy rose to **99.38% ± 0.39**.

**After fine-tuning with a 96-example replay buffer:** Old-task accuracy was **94.14% ± 3.77**, with **99.12% ± 0.16** on the new task.

**After frozen-base expansion and teacher distillation:** Old-task accuracy remained **99.79% ± 0.08**, and new-task accuracy reached **99.10% ± 0.42**.

**After additional synthetic dream rehearsal:** Old-task accuracy remained **99.79% ± 0.08**, but new-task accuracy declined slightly to **98.79% ± 0.57**.

The experiment establishes a narrow but concrete result: on this toy setup, naive training interferes with prior behavior, rehearsal reduces that interference, and a properly routed frozen-base adapter can acquire another task without altering old-task outputs.

It also establishes that this particular pseudo-label rehearsal step did not improve performance after successful distillation.

What it does **not** establish is equally important.

It does not demonstrate LLM-scale continual learning, semantic knowledge consolidation, autonomous discovery of task boundaries, a successful learned MoE router, truly on-policy sequence distillation, or an implementation of the authors' policy-learning dream generator.

The exact preservation of A is an architectural property of the experiment, not an emergent property learned by the model.

Three seeds and simple Gaussian tasks are not enough for sweeping conclusions. The confidence intervals are not estimates of what will happen on other architectures or datasets. Use the JSON output as a reproducible starting point, not as a benchmark against Google Research.

### Run your own ablations

The companion project includes the full code, a README, a JSON output file, and unit tests.

Start by changing the replay buffer from 96 examples to 16, 32, and 256. Measure old-task retention, new-task accuracy, and training time.

Then change the adapter rank from eight to two, four, sixteen, and thirty-two. Does new-task accuracy plateau? Does more capacity help beyond the complexity needed by the simple decision boundary?

Then remove the task-ID feature and replace the hard-coded gate with a learned router. This is the first ablation that removes a major crutch from our demonstration.

Next, make Task B partially conflict with Task A, introduce ambiguous routing, and evaluate sequentially across four or more tasks.

Finally, measure whether synthetic rehearsal helps before the adapter has converged. Dream rehearsal after a near-perfect solution may be an especially unfavorable place to evaluate its value.

Every one of these experiments can teach us something. None turns this toy directly into a reproduction of the full Sleep paper.

## 19. How to Design a More Faithful Replication

A research-grade replication would require a very different engineering plan.

First, select a specific base model and control its tokenizer, training configuration, context window, and generation policy. Choose a model small enough to fine-tune reproducibly but large enough for meaningful sequential-learning experiments.

Second, implement at least two memory components with genuinely different update frequencies, including a fast temporary state and a slower persistent module. Simply adding a LoRA adapter to a frozen model does not create the same memory hierarchy.

Third, implement the paper's ordered consolidation. The student must incorporate the prospective state of an update while the teacher reflects the pre-update state. Only the expanded capacity should train during that consolidation phase. Temporary capacity should be reset only after the transfer has been evaluated.

Fourth, use actual teacher and student sequence distributions. Generate some sequences from the student, score them with the teacher at corresponding token positions, and combine this with the teacher-sampled component described by GKD.

Fifth, add a proper LTI reward. It should evaluate both semantic correctness and trajectory similarity. The semantic reward model must be held fixed during the relevant optimization step, and the reward must not silently leak the held-out answer into generation.

Sixth, implement Dreaming with synthetic generation, controlled expert perturbations, gradient-based selection, isolated adaptation trials, and a reward derived from genuine downstream improvement. In a computationally practical setup, cache or approximate expensive inner-loop evaluations and report any approximation explicitly.

Seventh, run an appropriate continual-learning benchmark with multiple task orders. At minimum, measure final average accuracy, per-task forgetting, forward transfer, backward transfer, adaptation latency, incremental memory allocation, total training FLOPs or GPU hours, and performance under ambiguous routing.

Eighth, compare against strong baselines with comparable resources: naive continued training, replay, EWC, fixed-capacity LoRA, separate adapters with a learned router, parameter isolation, retrieval augmentation, and on-policy distillation without memory expansion.

Ninth, report confidence intervals and sensitivity to seeds, dataset order, update frequency, adapter rank, teacher sampling policy, and number of generated dreams. A single favorable task ordering is not enough.

Tenth, use a strict holdout policy. If the dream generator, reward function, or data-selection procedure can see evaluation answers, the apparent learning gain may be a disguised data leak.

The full architecture deserves this kind of evaluation because its claim is not merely that it can fit a new batch of data. Its claim is that a repeatable learning lifecycle can accumulate knowledge while preserving useful prior abilities.

## 20. What Would Make This Practical in Deployed LLMs?

Suppose a coding assistant watches an organization ship hundreds of new APIs, migrate databases, revise runbooks, and adopt internal style conventions. A continual-learning system could eventually consolidate recurring patterns into parameters and become more helpful without continuously carrying an enormous reference context.

That prospect is attractive. It introduces several problems that a laboratory benchmark cannot fully solve.

### When should the model sleep?

Frequent consolidation costs compute and may interrupt latency-sensitive work. Rare consolidation allows fast memory to become overloaded or overwritten. The right schedule likely depends on observed data drift, memory saturation, and the value of incoming information.

### What deserves durable memory?

Not every conversation should become a parameter update. Much of it is wrong, sensitive, transient, or specific to one person. An operational system needs consent, provenance, validation, retention policy, and methods to prevent contaminated experiences from becoming widely used behavior.

### How do we roll back a bad sleep cycle?

Production deployments need checkpoints, regression tests, canary evaluation, reproducible model artifacts, and a clear rule for promoting a newly consolidated checkpoint. Self-modifying weights without an auditable rollback path create a difficult reliability problem.

### Can memory be deleted?

Parametric learning is not a substitute for a deletion-capable database. A system that incorporates confidential or revoked facts into weights may struggle to remove them precisely. This matters for data governance and compliance.

### How does the model know which expert to use?

Our toy receives the task identifier for free. A production system must learn routing signals, infer context, handle ambiguity, and recover from selecting the wrong memory. A router can become a new source of forgetting even if every expert retains its content perfectly.

### What happens when synthetic data are wrong?

A model can reinforce its own mistakes through repeated self-training. Reward quality, source verification, uncertainty estimation, diversity, and independent holdout evaluation are therefore central, not optional.

### How much storage and compute are acceptable?

Expandable memory introduces a tension between total capacity and active capacity. An MoE system can keep active computation controlled while total parameters and associated optimizer state grow. Training, evaluation, persistence, and distribution still incur costs.

### What happens to the base model's safety and calibration?

Continual training may modify refusal behavior, uncertainty calibration, tool-use policies, or domain-specific safety boundaries. Those behaviors must be included in regression suites, not treated as separate from technical performance.

### Can external retrieval and parametric sleep complement each other?

Probably, in many designs. A retrievable source of truth provides explicit citations and update control. Consolidated parameters may offer fast access to reusable patterns. A practical system could decide which information belongs in external storage, which belongs in ephemeral context, and which is worth a more expensive parametric update.

That division of responsibility may be more robust than trying to encode every new fact directly into model weights.

## 21. Conclusion: The Important Idea Is Not Sleep. It Is Controlled Memory Transfer.

The Sleep framework asks an unusually useful question about language-model architecture.

If a model can adapt during a conversation, why should everything learned during that conversation have to disappear when the context ends? And if we want to preserve it through weight updates, how do we avoid destroying what was already learned?

The paper's answer has three intertwined parts: organize memory into components with different update rates, expand capacity before fragile knowledge is overwritten, and combine consolidation with carefully selected synthetic practice.

The full proposal is considerably more sophisticated than simply fine-tuning an LLM overnight. It contains an ordering protocol for memory updates, teacher-student distillation, imitation-based training signals, and a self-improvement loop that tries to measure whether its own practice examples are useful.

Our PyTorch experiment demonstrates a limited version of the underlying engineering tradeoff. Naive sequential training forgets. Replay helps. Frozen new capacity, when routed explicitly to the appropriate task, can learn new behavior while preserving old predictions. And additional synthetic rehearsal can make things slightly worse rather than better.

That last outcome is worth remembering.

Continual learning is not just a problem of finding more data or running more gradient steps. It is a problem of deciding **what to preserve, where to store it, when to transfer it, and how to verify that the transfer improved the system instead of quietly damaging it**.

A model does not need to resemble a sleeping human for this insight to matter.

It needs a reliable way to learn without treating yesterday's knowledge as disposable.

---

## 22. References

**[1] Original research paper.** Ali Behrouz, Farnoosh Hashemi, Adel Javanmard, and Vahab Mirrokni. *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.* arXiv:2606.03979, revised July 10, 2026. Primary source for the architecture, objectives, experimental settings, and reported benchmark numbers.

https://arxiv.org/html/2606.03979

**[2] Google Research.** Ali Behrouz and Vahab Mirrokni. *The sleep framework: A new way for LLMs to retain knowledge in continual learning.* October 27, 2026. Research overview and explanation of memory consolidation and dreaming.

https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/

**[3] Nested Learning and Hope.** *Introducing Nested Learning: A new ML paradigm for continual learning.* Google Research, 2025. Background on multi-timescale memory and the Hope model family.

https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/

**[4] GKD and on-policy distillation.** Rishabh Agarwal and collaborators. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.* ICLR 2024.

https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html

**[5] Self-Adapting Language Models (SEAL).** Adam Zweiger and collaborators. *Self-Adapting Language Models.* NeurIPS 2025. Primary reference for self-generated training data and reward-based adaptation.

https://arxiv.org/abs/2506.10943

**[6] BABILong.** Yuri Kuratov and collaborators. *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack.* NeurIPS 2024.

https://arxiv.org/abs/2406.10149

**[7] Related but separate Sleep research.** Sangyun Lee, Sean McLeish, Tom Goldstein, and Giulia Fanti. *Language Models Need Sleep.* arXiv:2605.26099, 2026. This similarly named work studies a different offline-recurrence mechanism and should not be conflated with Google's Knowledge Seeding and Dreaming paper.

https://arxiv.org/abs/2605.26099

**[8] LoRA.** Edward J. Hu and collaborators. *LoRA: Low-Rank Adaptation of Large Language Models.* 2021. Background for low-rank trainable updates.

https://arxiv.org/abs/2106.09685

**[9] Elastic Weight Consolidation.** James Kirkpatrick and collaborators. *Overcoming catastrophic forgetting in neural networks.* PNAS, 2017.

https://doi.org/10.1073/pnas.1611835114

**[10] Companion experiment.** Independent PyTorch continual-learning laboratory accompanying this article. Includes synthetic tasks, naive and replay baselines, low-rank distillation, synthetic rehearsal, benchmark output, and unit tests. This code was independently written and tested; it is not an official or faithful implementation of Google's Sleep architecture.

*Methods note, October 30, 2026: We consulted the July 2026 revision of the paper and Google's October 27 research post. The three-seed PyTorch benchmark and seven unit tests were run locally on CPU. Paper results and our toy results are labeled separately throughout. No original Google Sleep model checkpoint or full training implementation was independently executed, and no claims of LLM-scale replication are made.*
