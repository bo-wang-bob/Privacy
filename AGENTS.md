# Repository guidance

This repository supports federated CLIP soft-prompt tuning with FedAvg, DP-FPL,
or FedASK and membership-privacy auditing.

- Do not reintroduce backdoor triggers, malicious-client poisoning, ASR metrics, or SEISMOGRAPH defenses.
- CLIP weights must be loaded locally with `local_files_only=True`.
- Only train and aggregate parameters whose `requires_grad` flag is true.
- Preserve detached tensor copies across clients and rounds.
- Lightweight tests must run without datasets or CLIP checkpoints.
- Do not commit `data/`, `results/`, `logs/`, `checkpoints/`, caches, or saved experiment models.
