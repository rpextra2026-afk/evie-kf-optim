## Table 9c -- divergence across tasks

Diverged cells per optimizer on each task, and the overall rate. Descriptive. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Optimizer | vis/cifar10 | vis/cifar100 | vis/stl10 | vis/svhn | lang/enwik8 | lang/ptb | lang/wikitext103 | lang/wikitext2 | fin/consumer_industrials | fin/finance | fin/healthcare | fin/tech | total | rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EvieKF | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2/1025 | 0.20% |
| AdamW | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1025 | 0.00% |
| AdaBelief | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0/1025 | 0.00% |
| SGD | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 4 | 4 | 2 | 13/1025 | 1.27% |
| Sophia | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 14 | 2 | 18/1025 | 1.76% |
| Muon | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 32 | 2 | 0 | 34/1025 | 3.32% |
| Lion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 5 | 8 | 30/1025 | 2.93% |
| Shampoo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 11 | 0 | 7 | 21/1025 | 2.05% |
