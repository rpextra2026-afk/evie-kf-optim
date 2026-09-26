## Table 1 -- main results

Primary metric per task: median over 5 seeds of the per-seed value (for multi-unit tasks, the median over complete-case units). Best per task in bold. vis/cifar10: test top-1 acc (%); vis/cifar100: test top-1 acc (%); vis/stl10: test top-1 acc (%); vis/svhn: test top-1 acc (%); lang/enwik8: test perplexity; lang/ptb: test perplexity; lang/wikitext103: test perplexity; lang/wikitext2: test perplexity; fin/consumer_industrials: test MSE x1e-4; fin/finance: test MSE x1e-4; fin/healthcare: test MSE x1e-4; fin/tech: test MSE x1e-4. Complete-case units per task: vis/cifar10 1/1, vis/cifar100 1/1, vis/stl10 1/1, vis/svhn 1/1, lang/enwik8 1/1, lang/ptb 1/1, lang/wikitext103 1/1, lang/wikitext2 1/1, fin/consumer_industrials 45/50, fin/finance 23/49, fin/healthcare 33/49, fin/tech 37/49 (Table 9b re-scores with failures counted). Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Optimizer | vis/cifar10 | vis/cifar100 | vis/stl10 | vis/svhn | lang/enwik8 | lang/ptb | lang/wikitext103 | lang/wikitext2 | fin/consumer_industrials | fin/finance | fin/healthcare | fin/tech |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EvieKF | 81.94 | 52.28 | 48.59 | 94.63 | 3.04 | 192.17 | 35.41 | 100.46 | 1.9687 | 2.2669 | 2.9150 | 4.1497 |
| AdamW | 81.55 | 49.51 | **51.29** | 94.66 | 3.18 | 179.35 | 39.99 | 105.70 | 2.0169 | 2.2643 | 3.0969 | 4.3336 |
| AdaBelief | 81.00 | 49.45 | 50.84 | 94.75 | 3.15 | 179.77 | 41.17 | 97.15 | 2.1041 | 2.2583 | 3.1291 | 4.3745 |
| SGD | 81.71 | **54.27** | 47.75 | 94.73 | 3.60 | 180.35 | 59.75 | 103.80 | **1.8491** | 2.2278 | **2.9077** | 4.1537 |
| Sophia | 81.52 | 47.39 | 47.48 | 94.89 | 3.34 | 201.49 | 39.51 | 123.75 | 2.0623 | 2.6356 | 3.1868 | 4.5594 |
| Muon | **85.86** | 54.24 | 48.85 | **95.52** | **2.97** | 187.01 | **33.49** | **83.62** | 1.9124 | 3.7243 | 3.2588 | **4.1475** |
| Lion | 81.28 | 49.59 | 49.96 | 94.55 | 3.11 | 201.94 | 47.31 | 106.65 | 2.1161 | **2.2269** | 3.1281 | 4.1502 |
| Shampoo | 81.90 | 50.80 | 51.25 | 94.83 | 2.97 | **172.94** | 36.25 | 123.91 | 2.0574 | 3.3812 | 3.1132 | 4.2170 |
