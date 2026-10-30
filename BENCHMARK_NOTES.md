# Benchmark interpretation

Captured October 30, 2026, using PyTorch 2.10.0+cpu.

This is a synthetic 2-task binary classifier benchmark, **not a language model**.

- Task A: label is first coordinate > 0.
- Task B: label is second coordinate > 0.
- Input gives the network task identity in its third feature.
- Training examples per task: 1400; holdout examples per task: 1400.
- Seeds: 0, 1, 2.
- Initial Task A training: 180 Adam steps; adaptation to B: 220 Adam steps.
- Replay baseline stores only 96 A training examples.
- Expanded model uses rank-8 B-only residual, with a frozen base and *perfect task gate* using the explicit input task feature.
- Distillation uses synthetic unlabeled B inputs and KL against B teacher predictions.
- Dream rehearsal uses pseudo-labels and output-disagreement importance proxy, not actual full parameter-gradient selection or RL.

Results are means and sample standard deviations over seeds 0, 1, and 2. See results/benchmark.json for individual runs. The exact A preservation for the expanded model is engineered by the input task gate; it is **not** evidence of robust learned routing. Dream rehearsal slightly lowered B accuracy in this experiment.

The code is an independent, deliberately simplified educational reconstruction of only certain ideas. No Google code or weights were executed, and this is not a replication of arXiv:2606.03979.
