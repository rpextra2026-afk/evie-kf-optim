## Table 2 -- rank per task

Rank of each optimizer on each task (1 = best) and mean rank. Descriptive summary only. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Optimizer | vis/cifar10 | vis/cifar100 | vis/stl10 | vis/svhn | lang/enwik8 | lang/ptb | lang/wikitext103 | lang/wikitext2 | fin/consumer_industrials | fin/finance | fin/healthcare | fin/tech | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EvieKF | 2 | 3 | 6 | 7 | 3 | 6 | 2 | 3 | 3 | 5 | 2 | 2 | 3.67 |
| AdamW | 5 | 6 | 1 | 6 | 6 | 2 | 5 | 5 | 4 | 4 | 3 | 6 | 4.42 |
| AdaBelief | 8 | 7 | 3 | 4 | 5 | 3 | 6 | 2 | 7 | 3 | 6 | 7 | 5.08 |
| SGD | 4 | 1 | 7 | 5 | 8 | 4 | 8 | 4 | 1 | 2 | 1 | 4 | 4.08 |
| Sophia | 6 | 8 | 8 | 2 | 7 | 7 | 4 | 7 | 6 | 6 | 7 | 8 | 6.33 |
| Muon | 1 | 2 | 5 | 1 | 1 | 5 | 1 | 1 | 2 | 8 | 8 | 1 | **3.00** |
| Lion | 7 | 5 | 4 | 8 | 4 | 8 | 7 | 6 | 8 | 1 | 5 | 3 | 5.50 |
| Shampoo | 3 | 4 | 2 | 3 | 2 | 1 | 3 | 8 | 5 | 7 | 4 | 5 | 3.92 |
