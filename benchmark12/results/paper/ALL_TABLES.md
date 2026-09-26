# Paper tables

Regenerate with `python benchmark12/results/build_paper_tables.py`. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech. Each table also exists as .tex and .csv.

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

## Table 3 -- robustness across seeds

Per-seed primary metric (seeds 42, 43, 44, 45, 46), with the median that enters Table 1, mean ± sample sd, and the spread (max-min)/min. Same units and scaling as Table 1. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Task | Optimizer | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | median | mean ± sd | spread |
|---|---|---|---|---|---|---|---|---|---|
| vis/cifar10 | EvieKF | 81.94 | 80.98 | 82.34 | 81.26 | 82.41 | 81.94 | 81.79 ± 0.64 | 1.77% |
| vis/cifar10 | AdamW | 82.35 | 81.16 | 81.55 | 81.13 | 81.79 | 81.55 | 81.60 ± 0.50 | 1.50% |
| vis/cifar10 | AdaBelief | 81.47 | 80.64 | 81.73 | 80.74 | 81.00 | 81.00 | 81.12 ± 0.47 | 1.35% |
| vis/cifar10 | SGD | 81.71 | 81.27 | 81.79 | 82.95 | 81.38 | 81.71 | 81.82 ± 0.67 | 2.07% |
| vis/cifar10 | Sophia | 81.28 | 81.52 | 81.45 | 81.91 | 82.27 | 81.52 | 81.69 ± 0.40 | 1.22% |
| vis/cifar10 | Muon | 85.78 | 86.07 | 85.86 | 86.03 | 85.74 | 85.86 | 85.90 ± 0.15 | 0.38% |
| vis/cifar10 | Lion | 81.48 | 81.28 | 81.42 | 80.55 | 80.66 | 81.28 | 81.08 ± 0.44 | 1.15% |
| vis/cifar10 | Shampoo | 81.95 | 79.33 | 81.71 | 82.12 | 81.90 | 81.90 | 81.40 ± 1.17 | 3.52% |
| vis/cifar100 | EvieKF | 52.34 | 52.65 | 52.28 | 51.85 | 51.50 | 52.28 | 52.12 ± 0.45 | 2.23% |
| vis/cifar100 | AdamW | 49.51 | 50.19 | 49.65 | 49.51 | 48.50 | 49.51 | 49.47 ± 0.61 | 3.48% |
| vis/cifar100 | AdaBelief | 49.45 | 49.30 | 49.83 | 49.60 | 48.61 | 49.45 | 49.36 ± 0.46 | 2.51% |
| vis/cifar100 | SGD | 54.46 | 54.23 | 54.27 | 53.82 | 54.74 | 54.27 | 54.30 ± 0.34 | 1.71% |
| vis/cifar100 | Sophia | 47.01 | 49.08 | 47.33 | 47.55 | 47.39 | 47.39 | 47.67 ± 0.81 | 4.40% |
| vis/cifar100 | Muon | 54.51 | 53.73 | 53.04 | 54.30 | 54.24 | 54.24 | 53.96 ± 0.59 | 2.77% |
| vis/cifar100 | Lion | 49.59 | 49.50 | 50.27 | 49.83 | 48.32 | 49.59 | 49.50 ± 0.72 | 4.04% |
| vis/cifar100 | Shampoo | 51.87 | 51.50 | 50.63 | 50.80 | 50.38 | 50.80 | 51.04 ± 0.62 | 2.96% |
| vis/stl10 | EvieKF | 52.08 | 46.94 | 50.08 | 48.23 | 48.59 | 48.59 | 49.18 ± 1.97 | 10.95% |
| vis/stl10 | AdamW | 51.29 | 50.17 | 50.51 | 51.31 | 51.35 | 51.29 | 50.93 ± 0.55 | 2.34% |
| vis/stl10 | AdaBelief | 50.26 | 50.19 | 50.84 | 51.56 | 52.40 | 50.84 | 51.05 ± 0.93 | 4.41% |
| vis/stl10 | SGD | 48.12 | 47.56 | 47.92 | 45.39 | 47.75 | 47.75 | 47.35 ± 1.12 | 6.03% |
| vis/stl10 | Sophia | 48.01 | 47.46 | 47.48 | 46.56 | 48.64 | 47.48 | 47.63 ± 0.77 | 4.46% |
| vis/stl10 | Muon | 49.14 | 48.85 | 48.85 | 50.05 | 47.34 | 48.85 | 48.84 ± 0.98 | 5.73% |
| vis/stl10 | Lion | 49.56 | 49.96 | 50.73 | 49.25 | 50.23 | 49.96 | 49.95 ± 0.57 | 2.99% |
| vis/stl10 | Shampoo | 51.25 | 51.42 | 51.02 | 50.79 | 52.69 | 51.25 | 51.43 ± 0.74 | 3.74% |
| vis/svhn | EvieKF | 94.80 | 94.43 | 94.63 | 94.63 | 94.78 | 94.63 | 94.65 ± 0.15 | 0.40% |
| vis/svhn | AdamW | 94.91 | 94.59 | 94.58 | 94.86 | 94.66 | 94.66 | 94.72 ± 0.16 | 0.35% |
| vis/svhn | AdaBelief | 94.90 | 94.75 | 94.70 | 94.83 | 94.63 | 94.75 | 94.76 ± 0.11 | 0.28% |
| vis/svhn | SGD | 94.75 | 94.68 | 94.64 | 94.79 | 94.73 | 94.73 | 94.72 ± 0.06 | 0.15% |
| vis/svhn | Sophia | 94.75 | 94.83 | 95.08 | 95.01 | 94.89 | 94.89 | 94.91 ± 0.13 | 0.35% |
| vis/svhn | Muon | 95.76 | 95.20 | 95.52 | 95.56 | 95.32 | 95.52 | 95.47 ± 0.22 | 0.59% |
| vis/svhn | Lion | 94.55 | 94.69 | 94.54 | 94.83 | 94.28 | 94.55 | 94.58 ± 0.20 | 0.58% |
| vis/svhn | Shampoo | 94.97 | 94.78 | 94.76 | 94.89 | 94.83 | 94.83 | 94.85 ± 0.09 | 0.22% |
| lang/enwik8 | EvieKF | 3.03 | 3.04 | 3.04 | 3.04 | 3.03 | 3.04 | 3.04 ± 0.00 | 0.39% |
| lang/enwik8 | AdamW | 3.17 | 3.17 | 3.18 | 3.19 | 3.18 | 3.18 | 3.18 ± 0.01 | 0.88% |
| lang/enwik8 | AdaBelief | 3.15 | 3.17 | 3.14 | 3.16 | 3.15 | 3.15 | 3.15 ± 0.01 | 0.81% |
| lang/enwik8 | SGD | 3.60 | 3.60 | 3.40 | 3.59 | 3.61 | 3.60 | 3.56 ± 0.09 | 6.19% |
| lang/enwik8 | Sophia | 3.35 | 3.37 | 3.34 | 3.30 | 3.28 | 3.34 | 3.33 ± 0.04 | 2.70% |
| lang/enwik8 | Muon | 2.95 | 2.95 | 2.97 | 2.98 | 2.98 | 2.97 | 2.97 ± 0.02 | 1.35% |
| lang/enwik8 | Lion | 3.10 | 3.11 | 3.10 | 3.11 | 3.11 | 3.11 | 3.11 ± 0.01 | 0.43% |
| lang/enwik8 | Shampoo | 2.97 | 2.96 | 2.98 | 3.04 | 2.97 | 2.97 | 2.98 ± 0.03 | 2.56% |
| lang/ptb | EvieKF | 197.32 | 192.17 | 190.80 | 197.75 | 177.29 | 192.17 | 191.07 ± 8.29 | 11.54% |
| lang/ptb | AdamW | 178.45 | 179.35 | 184.23 | 183.72 | 177.53 | 179.35 | 180.66 ± 3.10 | 3.77% |
| lang/ptb | AdaBelief | 178.44 | 177.72 | 182.48 | 183.56 | 179.77 | 179.77 | 180.39 ± 2.54 | 3.29% |
| lang/ptb | SGD | 175.12 | 182.79 | 175.45 | 180.35 | 184.59 | 180.35 | 179.66 ± 4.27 | 5.41% |
| lang/ptb | Sophia | 195.28 | 201.84 | 203.22 | 201.49 | 197.32 | 201.49 | 199.83 ± 3.37 | 4.07% |
| lang/ptb | Muon | 187.01 | 194.55 | 184.80 | 191.88 | 186.88 | 187.01 | 189.02 ± 4.04 | 5.28% |
| lang/ptb | Lion | 187.29 | 201.94 | 207.56 | 207.54 | 192.54 | 201.94 | 199.37 ± 9.12 | 10.82% |
| lang/ptb | Shampoo | 172.94 | 176.07 | 173.56 | 171.10 | 172.46 | 172.94 | 173.23 ± 1.83 | 2.91% |
| lang/wikitext103 | EvieKF | 35.70 | 35.19 | 35.57 | 35.41 | 35.30 | 35.41 | 35.43 ± 0.21 | 1.46% |
| lang/wikitext103 | AdamW | 40.78 | 40.61 | 39.99 | 39.80 | 39.82 | 39.99 | 40.20 ± 0.46 | 2.46% |
| lang/wikitext103 | AdaBelief | 42.16 | 39.66 | 42.32 | 41.17 | 39.65 | 41.17 | 40.99 ± 1.30 | 6.75% |
| lang/wikitext103 | SGD | 59.75 | 56.64 | 60.62 | 56.00 | 59.83 | 59.75 | 58.57 ± 2.09 | 8.25% |
| lang/wikitext103 | Sophia | 39.37 | 41.12 | 39.95 | 39.26 | 39.51 | 39.51 | 39.84 ± 0.76 | 4.75% |
| lang/wikitext103 | Muon | 33.58 | 33.57 | 33.47 | 33.49 | 33.47 | 33.49 | 33.51 ± 0.05 | 0.33% |
| lang/wikitext103 | Lion | 49.35 | 41.66 | 47.31 | 48.68 | 43.64 | 47.31 | 46.13 ± 3.33 | 18.45% |
| lang/wikitext103 | Shampoo | 36.58 | 36.16 | 36.05 | 36.26 | 36.25 | 36.25 | 36.26 ± 0.20 | 1.46% |
| lang/wikitext2 | EvieKF | 99.67 | 98.33 | 102.56 | 100.46 | 101.59 | 100.46 | 100.52 ± 1.65 | 4.30% |
| lang/wikitext2 | AdamW | 104.85 | 105.86 | 106.35 | 103.75 | 105.70 | 105.70 | 105.30 ± 1.02 | 2.51% |
| lang/wikitext2 | AdaBelief | 94.80 | 97.15 | 98.15 | 97.08 | 97.85 | 97.15 | 97.01 ± 1.31 | 3.54% |
| lang/wikitext2 | SGD | 103.80 | 103.18 | 102.94 | 104.25 | 104.00 | 103.80 | 103.63 ± 0.55 | 1.27% |
| lang/wikitext2 | Sophia | 119.38 | 124.60 | 118.04 | 123.75 | 124.56 | 123.75 | 122.07 ± 3.12 | 5.56% |
| lang/wikitext2 | Muon | 84.41 | 82.92 | 83.62 | 84.32 | 83.18 | 83.62 | 83.69 ± 0.66 | 1.80% |
| lang/wikitext2 | Lion | 110.09 | 104.44 | 111.11 | 106.65 | 106.13 | 106.65 | 107.68 ± 2.81 | 6.39% |
| lang/wikitext2 | Shampoo | 125.52 | 123.05 | 123.91 | 126.42 | 119.38 | 123.91 | 123.66 ± 2.73 | 5.90% |
| fin/consumer_industrials | EvieKF | 2.0671 | 1.9108 | 1.9687 | 2.0176 | 1.9093 | 1.9687 | 1.9747 ± 0.0685 | 8.26% |
| fin/consumer_industrials | AdamW | 1.8502 | 2.0169 | 2.0184 | 1.8934 | 2.1125 | 2.0169 | 1.9783 ± 0.1058 | 14.18% |
| fin/consumer_industrials | AdaBelief | 1.9129 | 2.2333 | 2.1041 | 2.0937 | 2.2186 | 2.1041 | 2.1125 ± 0.1285 | 16.75% |
| fin/consumer_industrials | SGD | 1.8491 | 1.8809 | 1.8491 | 1.8491 | 1.8764 | 1.8491 | 1.8609 ± 0.0162 | 1.72% |
| fin/consumer_industrials | Sophia | 1.8491 | 2.2290 | 2.0751 | 2.0121 | 2.0623 | 2.0623 | 2.0455 ± 0.1365 | 20.55% |
| fin/consumer_industrials | Muon | 1.9124 | 1.8481 | 1.9297 | 1.9345 | 1.8639 | 1.9124 | 1.8977 ± 0.0394 | 4.68% |
| fin/consumer_industrials | Lion | 1.8945 | 2.2529 | 2.1161 | 2.1313 | 1.9985 | 2.1161 | 2.0787 ± 0.1368 | 18.92% |
| fin/consumer_industrials | Shampoo | 2.0521 | 2.0574 | 2.0137 | 2.0736 | 2.2052 | 2.0574 | 2.0804 ± 0.0732 | 9.51% |
| fin/finance | EvieKF | 2.2674 | 2.2357 | 2.2669 | 2.2458 | 2.2846 | 2.2669 | 2.2601 ± 0.0193 | 2.19% |
| fin/finance | AdamW | 2.2643 | 2.2658 | 2.2534 | 2.2498 | 2.2677 | 2.2643 | 2.2602 ± 0.0080 | 0.79% |
| fin/finance | AdaBelief | 2.2370 | 2.2958 | 2.2582 | 2.3863 | 2.2583 | 2.2583 | 2.2871 ± 0.0594 | 6.68% |
| fin/finance | SGD | 2.2227 | 2.2278 | 2.2380 | 2.2352 | 2.2276 | 2.2278 | 2.2303 ± 0.0062 | 0.69% |
| fin/finance | Sophia | 2.9367 | 3.3865 | 2.6356 | 2.4235 | 2.3531 | 2.6356 | 2.7471 ± 0.4235 | 43.92% |
| fin/finance | Muon | 3.6352 | 3.7243 | 6.4050 | 3.4128 | 5.6520 | 3.7243 | 4.5659 ± 1.3662 | 87.68% |
| fin/finance | Lion | 2.2345 | 2.2175 | 2.2289 | 2.2260 | 2.2269 | 2.2269 | 2.2268 ± 0.0061 | 0.77% |
| fin/finance | Shampoo | 2.7708 | 3.4165 | 3.3812 | 2.5820 | 3.4064 | 3.3812 | 3.1114 ± 0.4029 | 32.32% |
| fin/healthcare | EvieKF | 2.8877 | 3.1721 | 2.9055 | 2.9150 | 2.9165 | 2.9150 | 2.9593 ± 0.1195 | 9.85% |
| fin/healthcare | AdamW | 3.1433 | 3.0685 | 2.9433 | 3.0969 | 3.1141 | 3.0969 | 3.0732 ± 0.0775 | 6.79% |
| fin/healthcare | AdaBelief | 3.1444 | 3.1291 | 2.9214 | 3.0952 | 3.1870 | 3.1291 | 3.0954 ± 0.1027 | 9.09% |
| fin/healthcare | SGD | 2.9093 | 2.9008 | 2.9200 | 2.9036 | 2.9077 | 2.9077 | 2.9083 ± 0.0074 | 0.66% |
| fin/healthcare | Sophia | 3.1368 | 3.1872 | 3.1876 | 3.1309 | 3.1868 | 3.1868 | 3.1659 ± 0.0293 | 1.81% |
| fin/healthcare | Muon | 3.2343 | 3.2461 | 3.2588 | 3.2805 | 3.3510 | 3.2588 | 3.2741 ± 0.0463 | 3.61% |
| fin/healthcare | Lion | 3.1281 | 2.9051 | 3.1303 | 2.9031 | 3.1494 | 3.1281 | 3.0432 ± 0.1273 | 8.48% |
| fin/healthcare | Shampoo | 3.1132 | 2.8975 | 3.1927 | 2.9356 | 3.2434 | 3.1132 | 3.0765 ± 0.1538 | 11.94% |
| fin/tech | EvieKF | 4.1427 | 4.1554 | 4.1497 | 4.1424 | 4.1718 | 4.1497 | 4.1524 ± 0.0121 | 0.71% |
| fin/tech | AdamW | 4.3336 | 4.3734 | 4.5192 | 4.2921 | 4.2422 | 4.3336 | 4.3521 ± 0.1053 | 6.53% |
| fin/tech | AdaBelief | 4.4760 | 4.5757 | 4.3745 | 4.3186 | 4.3601 | 4.3745 | 4.4210 ± 0.1041 | 5.95% |
| fin/tech | SGD | 4.1508 | 4.1537 | 4.2352 | 4.2335 | 4.1503 | 4.1537 | 4.1847 ± 0.0454 | 2.05% |
| fin/tech | Sophia | 4.3045 | 4.5673 | 4.5978 | 4.3711 | 4.5594 | 4.5594 | 4.4800 ± 0.1327 | 6.81% |
| fin/tech | Muon | 4.1475 | 4.1462 | 4.1483 | 4.1471 | 4.1496 | 4.1475 | 4.1477 ± 0.0013 | 0.08% |
| fin/tech | Lion | 4.1571 | 4.1502 | 4.1463 | 4.1675 | 4.1465 | 4.1502 | 4.1536 ± 0.0090 | 0.51% |
| fin/tech | Shampoo | 4.5839 | 4.1430 | 4.3735 | 4.1623 | 4.2170 | 4.2170 | 4.2959 ± 0.1847 | 10.64% |

## Table 3b -- sensitivity: excluding the LR-search seed (42)

The learning-rate search runs at seed 42, and the batch order is determined by the seed alone, so the search saw exactly the data order of main-run seed 42 and each frozen LR is tuned to it. This table recomputes every task number as the median over the other seeds only. So far seed 42 was the best of the seeds in 25 of 96 optimizer-task cases (chance: 19.2). Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Task | Optimizer | all seeds | excl. 42 | change | rank (all) | rank (excl. 42) |
|---|---|---|---|---|---|---|
| vis/cifar10 | EvieKF | 81.94 | 81.80 | -0.17% | 2 | 3 |
| vis/cifar10 | AdamW | 81.55 | 81.35 | -0.24% | 5 | 6 |
| vis/cifar10 | AdaBelief | 81.00 | 80.87 | -0.16% | 8 | 8 |
| vis/cifar10 | SGD | 81.71 | 81.59 | -0.15% | 4 | 5 |
| vis/cifar10 | Sophia | 81.52 | 81.72 | +0.24% | 6 | 4 |
| vis/cifar10 | Muon | 85.86 | 85.94 | +0.10% | 1 | 1 |
| vis/cifar10 | Lion | 81.28 | 80.97 | -0.38% | 7 | 7 |
| vis/cifar10 | Shampoo | 81.90 | 81.81 | -0.12% | 3 | 2 |
| vis/cifar100 | EvieKF | 52.28 | 52.06 | -0.41% | 3 | 3 |
| vis/cifar100 | AdamW | 49.51 | 49.58 | +0.14% | 6 | 6 |
| vis/cifar100 | AdaBelief | 49.45 | 49.45 | +0.00% | 7 | 7 |
| vis/cifar100 | SGD | 54.27 | 54.25 | -0.04% | 1 | 1 |
| vis/cifar100 | Sophia | 47.39 | 47.47 | +0.17% | 8 | 8 |
| vis/cifar100 | Muon | 54.24 | 53.98 | -0.47% | 2 | 2 |
| vis/cifar100 | Lion | 49.59 | 49.66 | +0.15% | 5 | 5 |
| vis/cifar100 | Shampoo | 50.80 | 50.72 | -0.17% | 4 | 4 |
| vis/stl10 | EvieKF | 48.59 | 48.41 | -0.37% | 6 | 6 |
| vis/stl10 | AdamW | 51.29 | 50.91 | -0.73% | 1 | 3 |
| vis/stl10 | AdaBelief | 50.84 | 51.20 | +0.71% | 3 | 2 |
| vis/stl10 | SGD | 47.75 | 47.66 | -0.20% | 7 | 7 |
| vis/stl10 | Sophia | 47.48 | 47.47 | -0.01% | 8 | 8 |
| vis/stl10 | Muon | 48.85 | 48.85 | +0.00% | 5 | 5 |
| vis/stl10 | Lion | 49.96 | 50.09 | +0.26% | 4 | 4 |
| vis/stl10 | Shampoo | 51.25 | 51.22 | -0.05% | 2 | 1 |
| vis/svhn | EvieKF | 94.63 | 94.63 | -0.00% | 7 | 6 |
| vis/svhn | AdamW | 94.66 | 94.63 | -0.04% | 6 | 7 |
| vis/svhn | AdaBelief | 94.75 | 94.73 | -0.03% | 4 | 4 |
| vis/svhn | SGD | 94.73 | 94.71 | -0.03% | 5 | 5 |
| vis/svhn | Sophia | 94.89 | 94.95 | +0.06% | 2 | 2 |
| vis/svhn | Muon | 95.52 | 95.42 | -0.10% | 1 | 1 |
| vis/svhn | Lion | 94.55 | 94.62 | +0.07% | 8 | 8 |
| vis/svhn | Shampoo | 94.83 | 94.81 | -0.03% | 3 | 3 |
| lang/enwik8 | EvieKF | 3.04 | 3.04 | +0.03% | 3 | 3 |
| lang/enwik8 | AdamW | 3.18 | 3.18 | +0.09% | 6 | 6 |
| lang/enwik8 | AdaBelief | 3.15 | 3.16 | +0.21% | 5 | 5 |
| lang/enwik8 | SGD | 3.60 | 3.59 | -0.19% | 8 | 8 |
| lang/enwik8 | Sophia | 3.34 | 3.32 | -0.62% | 7 | 7 |
| lang/enwik8 | Muon | 2.97 | 2.98 | +0.24% | 1 | 1 |
| lang/enwik8 | Lion | 3.11 | 3.11 | +0.01% | 4 | 4 |
| lang/enwik8 | Shampoo | 2.97 | 2.98 | +0.21% | 2 | 2 |
| lang/ptb | EvieKF | 192.17 | 191.49 | -0.36% | 6 | 6 |
| lang/ptb | AdamW | 179.35 | 181.54 | +1.22% | 2 | 3 |
| lang/ptb | AdaBelief | 179.77 | 181.12 | +0.75% | 3 | 2 |
| lang/ptb | SGD | 180.35 | 181.57 | +0.68% | 4 | 4 |
| lang/ptb | Sophia | 201.49 | 201.66 | +0.09% | 7 | 7 |
| lang/ptb | Muon | 187.01 | 189.38 | +1.27% | 5 | 5 |
| lang/ptb | Lion | 201.94 | 204.74 | +1.39% | 8 | 8 |
| lang/ptb | Shampoo | 172.94 | 173.01 | +0.04% | 1 | 1 |
| lang/wikitext103 | EvieKF | 35.41 | 35.35 | -0.15% | 2 | 2 |
| lang/wikitext103 | AdamW | 39.99 | 39.90 | -0.22% | 5 | 5 |
| lang/wikitext103 | AdaBelief | 41.17 | 40.41 | -1.83% | 6 | 6 |
| lang/wikitext103 | SGD | 59.75 | 58.23 | -2.53% | 8 | 8 |
| lang/wikitext103 | Sophia | 39.51 | 39.73 | +0.56% | 4 | 4 |
| lang/wikitext103 | Muon | 33.49 | 33.48 | -0.03% | 1 | 1 |
| lang/wikitext103 | Lion | 47.31 | 45.47 | -3.88% | 7 | 7 |
| lang/wikitext103 | Shampoo | 36.25 | 36.20 | -0.13% | 3 | 3 |
| lang/wikitext2 | EvieKF | 100.46 | 101.02 | +0.56% | 3 | 3 |
| lang/wikitext2 | AdamW | 105.70 | 105.78 | +0.08% | 5 | 5 |
| lang/wikitext2 | AdaBelief | 97.15 | 97.50 | +0.36% | 2 | 2 |
| lang/wikitext2 | SGD | 103.80 | 103.59 | -0.20% | 4 | 4 |
| lang/wikitext2 | Sophia | 123.75 | 124.16 | +0.33% | 7 | 8 |
| lang/wikitext2 | Muon | 83.62 | 83.40 | -0.27% | 1 | 1 |
| lang/wikitext2 | Lion | 106.65 | 106.39 | -0.24% | 6 | 6 |
| lang/wikitext2 | Shampoo | 123.91 | 123.48 | -0.35% | 8 | 7 |
| fin/consumer_industrials | EvieKF | 1.9687 | 1.9398 | -1.47% | 3 | 3 |
| fin/consumer_industrials | AdamW | 2.0169 | 2.0176 | +0.04% | 4 | 4 |
| fin/consumer_industrials | AdaBelief | 2.1041 | 2.1613 | +2.72% | 7 | 8 |
| fin/consumer_industrials | SGD | 1.8491 | 1.8628 | +0.74% | 1 | 1 |
| fin/consumer_industrials | Sophia | 2.0623 | 2.0687 | +0.31% | 6 | 6 |
| fin/consumer_industrials | Muon | 1.9124 | 1.8968 | -0.82% | 2 | 2 |
| fin/consumer_industrials | Lion | 2.1161 | 2.1237 | +0.36% | 8 | 7 |
| fin/consumer_industrials | Shampoo | 2.0574 | 2.0655 | +0.39% | 5 | 5 |
| fin/finance | EvieKF | 2.2669 | 2.2564 | -0.46% | 5 | 3 |
| fin/finance | AdamW | 2.2643 | 2.2596 | -0.21% | 4 | 4 |
| fin/finance | AdaBelief | 2.2583 | 2.2770 | +0.83% | 3 | 5 |
| fin/finance | SGD | 2.2278 | 2.2315 | +0.17% | 2 | 2 |
| fin/finance | Sophia | 2.6356 | 2.5295 | -4.03% | 6 | 6 |
| fin/finance | Muon | 3.7243 | 4.6882 | +25.88% | 8 | 8 |
| fin/finance | Lion | 2.2269 | 2.2265 | -0.02% | 1 | 1 |
| fin/finance | Shampoo | 3.3812 | 3.3938 | +0.37% | 7 | 7 |
| fin/healthcare | EvieKF | 2.9150 | 2.9157 | +0.03% | 2 | 2 |
| fin/healthcare | AdamW | 3.0969 | 3.0827 | -0.46% | 3 | 5 |
| fin/healthcare | AdaBelief | 3.1291 | 3.1122 | -0.54% | 6 | 6 |
| fin/healthcare | SGD | 2.9077 | 2.9057 | -0.07% | 1 | 1 |
| fin/healthcare | Sophia | 3.1868 | 3.1870 | +0.01% | 7 | 7 |
| fin/healthcare | Muon | 3.2588 | 3.2697 | +0.33% | 8 | 8 |
| fin/healthcare | Lion | 3.1281 | 3.0177 | -3.53% | 5 | 3 |
| fin/healthcare | Shampoo | 3.1132 | 3.0642 | -1.57% | 4 | 4 |
| fin/tech | EvieKF | 4.1497 | 4.1526 | +0.07% | 2 | 3 |
| fin/tech | AdamW | 4.3336 | 4.3328 | -0.02% | 6 | 6 |
| fin/tech | AdaBelief | 4.3745 | 4.3673 | -0.17% | 7 | 7 |
| fin/tech | SGD | 4.1537 | 4.1936 | +0.96% | 4 | 5 |
| fin/tech | Sophia | 4.5594 | 4.5633 | +0.09% | 8 | 8 |
| fin/tech | Muon | 4.1475 | 4.1477 | +0.00% | 1 | 1 |
| fin/tech | Lion | 4.1502 | 4.1484 | -0.04% | 3 | 2 |
| fin/tech | Shampoo | 4.2170 | 4.1896 | -0.65% | 5 | 4 |

## Table 4 -- EvieKF against each baseline

Positive = EvieKF better. 'task-level' compares the Table 1 numbers. 'paired median' pairs EvieKF with the baseline unit by unit (complete-case tickers) for multi-unit tasks, or seed by seed otherwise, and takes the median relative difference; 'EvieKF wins' counts those pairs. Descriptive: no p-values, because with many paired units a negligible gap is still 'significant'. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Task | Baseline | task-level Δ | paired median Δ | EvieKF wins |
|---|---|---|---|---|
| vis/cifar10 | AdamW | +0.48% | +0.160% | 3/5 seeds |
| vis/cifar10 | AdaBelief | +1.16% | +0.644% | 5/5 seeds |
| vis/cifar10 | SGD | +0.28% | +0.281% | 3/5 seeds |
| vis/cifar10 | Sophia | +0.52% | +0.170% | 3/5 seeds |
| vis/cifar10 | Muon | -4.57% | -4.477% | 0/5 seeds |
| vis/cifar10 | Lion | +0.81% | +0.881% | 4/5 seeds |
| vis/cifar10 | Shampoo | +0.05% | +0.623% | 3/5 seeds |
| vis/cifar100 | AdamW | +5.59% | +5.297% | 5/5 seeds |
| vis/cifar100 | AdaBelief | +5.72% | +5.844% | 5/5 seeds |
| vis/cifar100 | SGD | -3.67% | -3.667% | 0/5 seeds |
| vis/cifar100 | Sophia | +10.32% | +9.043% | 5/5 seeds |
| vis/cifar100 | Muon | -3.61% | -3.981% | 0/5 seeds |
| vis/cifar100 | Lion | +5.42% | +5.545% | 5/5 seeds |
| vis/cifar100 | Shampoo | +2.91% | +2.223% | 5/5 seeds |
| vis/stl10 | AdamW | -5.26% | -5.380% | 1/5 seeds |
| vis/stl10 | AdaBelief | -4.43% | -6.473% | 1/5 seeds |
| vis/stl10 | SGD | +1.75% | +4.486% | 4/5 seeds |
| vis/stl10 | Sophia | +2.34% | +3.570% | 3/5 seeds |
| vis/stl10 | Muon | -0.54% | +2.508% | 3/5 seeds |
| vis/stl10 | Lion | -2.75% | -2.081% | 1/5 seeds |
| vis/stl10 | Shampoo | -5.20% | -5.046% | 1/5 seeds |
| vis/svhn | AdamW | -0.03% | -0.117% | 2/5 seeds |
| vis/svhn | AdaBelief | -0.13% | -0.105% | 1/5 seeds |
| vis/svhn | SGD | -0.11% | -0.012% | 2/5 seeds |
| vis/svhn | Sophia | -0.27% | -0.392% | 1/5 seeds |
| vis/svhn | Muon | -0.93% | -0.933% | 0/5 seeds |
| vis/svhn | Lion | +0.09% | +0.098% | 3/5 seeds |
| vis/svhn | Shampoo | -0.21% | -0.174% | 0/5 seeds |
| lang/enwik8 | AdamW | +4.33% | +4.330% | 5/5 seeds |
| lang/enwik8 | AdaBelief | +3.49% | +3.693% | 5/5 seeds |
| lang/enwik8 | SGD | +15.62% | +15.638% | 5/5 seeds |
| lang/enwik8 | Sophia | +9.07% | +9.072% | 5/5 seeds |
| lang/enwik8 | Muon | -2.36% | -2.360% | 0/5 seeds |
| lang/enwik8 | Lion | +2.23% | +2.204% | 5/5 seeds |
| lang/enwik8 | Shampoo | -2.27% | -2.104% | 0/5 seeds |
| lang/ptb | AdamW | -7.15% | -7.152% | 1/5 seeds |
| lang/ptb | AdaBelief | -6.90% | -7.731% | 1/5 seeds |
| lang/ptb | SGD | -6.55% | -8.749% | 1/5 seeds |
| lang/ptb | Sophia | +4.62% | +4.789% | 4/5 seeds |
| lang/ptb | Muon | -2.76% | -3.061% | 2/5 seeds |
| lang/ptb | Lion | +4.84% | +4.836% | 4/5 seeds |
| lang/ptb | Shampoo | -11.12% | -9.935% | 0/5 seeds |
| lang/wikitext103 | AdamW | +11.47% | +11.350% | 5/5 seeds |
| lang/wikitext103 | AdaBelief | +13.99% | +13.994% | 5/5 seeds |
| lang/wikitext103 | SGD | +40.74% | +40.244% | 5/5 seeds |
| lang/wikitext103 | Sophia | +10.39% | +10.664% | 5/5 seeds |
| lang/wikitext103 | Muon | -5.72% | -5.722% | 0/5 seeds |
| lang/wikitext103 | Lion | +25.16% | +24.816% | 5/5 seeds |
| lang/wikitext103 | Shampoo | +2.32% | +2.404% | 5/5 seeds |
| lang/wikitext2 | AdamW | +4.96% | +3.891% | 5/5 seeds |
| lang/wikitext2 | AdaBelief | -3.40% | -3.819% | 0/5 seeds |
| lang/wikitext2 | SGD | +3.22% | +3.636% | 5/5 seeds |
| lang/wikitext2 | Sophia | +18.82% | +18.445% | 5/5 seeds |
| lang/wikitext2 | Muon | -20.13% | -19.144% | 0/5 seeds |
| lang/wikitext2 | Lion | +5.80% | +5.847% | 5/5 seeds |
| lang/wikitext2 | Shampoo | +18.93% | +20.088% | 5/5 seeds |
| fin/consumer_industrials | AdamW | +2.39% | +0.124% | 23/45 units |
| fin/consumer_industrials | AdaBelief | +6.43% | +0.349% | 24/45 units |
| fin/consumer_industrials | SGD | -6.47% | -1.052% | 8/45 units |
| fin/consumer_industrials | Sophia | +4.54% | +0.343% | 26/45 units |
| fin/consumer_industrials | Muon | -2.94% | -0.489% | 17/45 units |
| fin/consumer_industrials | Lion | +6.96% | +1.007% | 29/45 units |
| fin/consumer_industrials | Shampoo | +4.31% | +0.179% | 24/45 units |
| fin/finance | AdamW | -0.11% | -0.171% | 9/23 units |
| fin/finance | AdaBelief | -0.38% | +1.615% | 17/23 units |
| fin/finance | SGD | -1.76% | -2.094% | 6/23 units |
| fin/finance | Sophia | +13.99% | +7.255% | 22/23 units |
| fin/finance | Muon | +39.13% | +37.304% | 22/23 units |
| fin/finance | Lion | -1.79% | -2.536% | 5/23 units |
| fin/finance | Shampoo | +32.96% | +13.831% | 23/23 units |
| fin/healthcare | AdamW | +5.87% | +0.344% | 21/33 units |
| fin/healthcare | AdaBelief | +6.84% | +0.428% | 22/33 units |
| fin/healthcare | SGD | -0.25% | -0.106% | 14/33 units |
| fin/healthcare | Sophia | +8.53% | -0.119% | 14/33 units |
| fin/healthcare | Muon | +10.55% | +2.218% | 26/33 units |
| fin/healthcare | Lion | +6.81% | -0.046% | 15/33 units |
| fin/healthcare | Shampoo | +6.37% | +0.051% | 18/33 units |
| fin/tech | AdamW | +4.25% | +2.893% | 31/37 units |
| fin/tech | AdaBelief | +5.14% | +2.616% | 31/37 units |
| fin/tech | SGD | +0.10% | -0.012% | 18/37 units |
| fin/tech | Sophia | +8.99% | +3.468% | 36/37 units |
| fin/tech | Muon | -0.05% | -0.021% | 10/37 units |
| fin/tech | Lion | +0.01% | -0.041% | 14/37 units |
| fin/tech | Shampoo | +1.60% | +0.018% | 22/37 units |

## Table 5 -- per-task relative improvement d_i

d_i = (ref - opt)/|ref| for lower-is-better metrics, sign-flipped otherwise; d_i > 0 means the optimizer beat the reference on that task. These are exactly the values benchmark12/aggregate.py feeds to the paired Wilcoxon across tasks (with Holm over 7 comparisons). The test is run twice, reference AdamW and reference EvieKF. NOT A RESULT until all 12 tasks exist -- Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Reference | Optimizer | vis/cifar10 | vis/cifar100 | vis/stl10 | vis/svhn | lang/enwik8 | lang/ptb | lang/wikitext103 | lang/wikitext2 | fin/consumer_industrials | fin/finance | fin/healthcare | fin/tech | beats ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AdamW | EvieKF | +0.0048 | +0.0559 | -0.0526 | -0.0003 | +0.0433 | -0.0715 | +0.1147 | +0.0496 | +0.0239 | -0.0011 | +0.0587 | +0.0425 | 8/12 |
| AdamW | AdaBelief | -0.0067 | -0.0012 | -0.0088 | +0.0009 | +0.0087 | -0.0024 | -0.0293 | +0.0809 | -0.0432 | +0.0027 | -0.0104 | -0.0094 | 4/12 |
| AdamW | SGD | +0.0020 | +0.0961 | -0.0690 | +0.0007 | -0.1338 | -0.0056 | -0.4939 | +0.0180 | +0.0832 | +0.0161 | +0.0611 | +0.0415 | 8/12 |
| AdamW | Sophia | -0.0004 | -0.0428 | -0.0743 | +0.0024 | -0.0522 | -0.1235 | +0.0121 | -0.1708 | -0.0225 | -0.1640 | -0.0290 | -0.0521 | 2/12 |
| AdamW | Muon | +0.0529 | +0.0955 | -0.0475 | +0.0090 | +0.0654 | -0.0427 | +0.1626 | +0.2089 | +0.0518 | -0.6448 | -0.0523 | +0.0430 | 8/12 |
| AdamW | Lion | -0.0033 | +0.0016 | -0.0258 | -0.0012 | +0.0214 | -0.1260 | -0.1829 | -0.0090 | -0.0492 | +0.0165 | -0.0101 | +0.0423 | 4/12 |
| AdamW | Shampoo | +0.0043 | +0.0261 | -0.0007 | +0.0018 | +0.0646 | +0.0357 | +0.0937 | -0.1723 | -0.0201 | -0.4933 | -0.0053 | +0.0269 | 7/12 |
| EvieKF | AdamW | -0.0048 | -0.0530 | +0.0556 | +0.0003 | -0.0453 | +0.0667 | -0.1296 | -0.0522 | -0.0245 | +0.0011 | -0.0624 | -0.0443 | 4/12 |
| EvieKF | AdaBelief | -0.0115 | -0.0541 | +0.0463 | +0.0013 | -0.0361 | +0.0645 | -0.1627 | +0.0329 | -0.0687 | +0.0038 | -0.0735 | -0.0542 | 5/12 |
| EvieKF | SGD | -0.0028 | +0.0381 | -0.0172 | +0.0011 | -0.1851 | +0.0615 | -0.6875 | -0.0332 | +0.0607 | +0.0173 | +0.0025 | -0.0010 | 6/12 |
| EvieKF | Sophia | -0.0051 | -0.0935 | -0.0229 | +0.0027 | -0.0998 | -0.0485 | -0.1159 | -0.2319 | -0.0475 | -0.1627 | -0.0933 | -0.0987 | 1/12 |
| EvieKF | Muon | +0.0478 | +0.0375 | +0.0054 | +0.0094 | +0.0231 | +0.0269 | +0.0541 | +0.1676 | +0.0286 | -0.6429 | -0.1180 | +0.0005 | 10/12 |
| EvieKF | Lion | -0.0081 | -0.0515 | +0.0283 | -0.0009 | -0.0229 | -0.0508 | -0.3362 | -0.0616 | -0.0748 | +0.0176 | -0.0731 | -0.0001 | 2/12 |
| EvieKF | Shampoo | -0.0005 | -0.0283 | +0.0548 | +0.0021 | +0.0222 | +0.1001 | -0.0238 | -0.2335 | -0.0450 | -0.4916 | -0.0680 | -0.0162 | 4/12 |

## Table 6 -- frozen hyperparameters and tuning budget

Frozen learning rate and second-axis knob (EvieKF: gamma; Shampoo: beta), and how much search produced them: grid points scored, training runs (points x LR-search units -- 8 tickers for finance, 1 otherwise), total search steps, and boundary extensions. EvieKF and Shampoo search a joint (lr, knob) grid; every other arm searches LR only, so the budget is NOT equal across arms -- this table is the disclosure. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Task | Optimizer | frozen lr | frozen knob | grid pts | search runs | search steps | lr extends | knob extends |
|---|---|---|---|---|---|---|---|---|
| vis/cifar10 | EvieKF | 0.09999 | 27000 | 99 | 99 | 247,500 | 2 | 3 |
| vis/cifar10 | AdamW | 0.01 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar10 | AdaBelief | 0.01 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar10 | SGD | 0.3 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar10 | Sophia | 0.003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar10 | Muon | 0.006325 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar10 | Lion | 0.003162 | -- | 9 | 9 | 22,500 | 1 | 0 |
| vis/cifar10 | Shampoo | 0.03162 | 0.95 | 36 | 36 | 90,000 | 1 | 0 |
| vis/cifar100 | EvieKF | 0.01 | 1000 | 49 | 49 | 122,500 | 0 | 0 |
| vis/cifar100 | AdamW | 0.003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar100 | AdaBelief | 0.003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar100 | SGD | 0.3 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar100 | Sophia | 0.001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar100 | Muon | 0.02 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar100 | Lion | 0.001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/cifar100 | Shampoo | 0.003162 | 0.85 | 42 | 42 | 105,000 | 0 | 1 |
| vis/stl10 | EvieKF | 0.03162 | 3 | 81 | 81 | 202,500 | 2 | 1 |
| vis/stl10 | AdamW | 0.01 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/stl10 | AdaBelief | 0.003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/stl10 | SGD | 0.3 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/stl10 | Sophia | 0.003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/stl10 | Muon | 0.02 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/stl10 | Lion | 0.003162 | -- | 9 | 9 | 22,500 | 1 | 0 |
| vis/stl10 | Shampoo | 0.003162 | 0.9 | 42 | 42 | 105,000 | 0 | 2 |
| vis/svhn | EvieKF | 0.01 | 10 | 49 | 49 | 122,500 | 0 | 0 |
| vis/svhn | AdamW | 0.003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/svhn | AdaBelief | 0.003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/svhn | SGD | 0.09487 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/svhn | Sophia | 0.003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/svhn | Muon | 0.006325 | -- | 7 | 7 | 17,500 | 0 | 0 |
| vis/svhn | Lion | 0.003162 | -- | 9 | 9 | 22,500 | 1 | 0 |
| vis/svhn | Shampoo | 0.003162 | 0.7 | 70 | 70 | 175,000 | 0 | 3 |
| lang/enwik8 | EvieKF | 0.0003162 | 27000 | 69 | 69 | 172,500 | 0 | 3 |
| lang/enwik8 | AdamW | 0.0003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/enwik8 | AdaBelief | 0.0003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/enwik8 | SGD | 0.09487 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/enwik8 | Sophia | 0.001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/enwik8 | Muon | 0.006325 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/enwik8 | Lion | 0.0001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/enwik8 | Shampoo | 0.003162 | 0.75 | 56 | 56 | 140,000 | 0 | 2 |
| lang/ptb | EvieKF | 0.001 | 10 | 49 | 49 | 122,500 | 0 | 0 |
| lang/ptb | AdamW | 0.0003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/ptb | AdaBelief | 0.0003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/ptb | SGD | 0.09487 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/ptb | Sophia | 0.0003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/ptb | Muon | 0.006325 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/ptb | Lion | 0.0001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/ptb | Shampoo | 0.001 | 0.99 | 28 | 28 | 70,000 | 0 | 0 |
| lang/wikitext103 | EvieKF | 0.001 | 243000 | 77 | 77 | 192,500 | 0 | 3 |
| lang/wikitext103 | AdamW | 0.001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext103 | AdaBelief | 0.001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext103 | SGD | 0.09487 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext103 | Sophia | 0.001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext103 | Muon | 0.002 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext103 | Lion | 0.0001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext103 | Shampoo | 0.003162 | 0.8 | 43 | 43 | 107,500 | 0 | 3 |
| lang/wikitext2 | EvieKF | 0.001 | 300 | 49 | 49 | 122,500 | 0 | 0 |
| lang/wikitext2 | AdamW | 0.001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext2 | AdaBelief | 0.0003162 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext2 | SGD | 0.09487 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext2 | Sophia | 0.001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext2 | Muon | 0.006325 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext2 | Lion | 0.0001 | -- | 7 | 7 | 17,500 | 0 | 0 |
| lang/wikitext2 | Shampoo | 0.003162 | 0.95 | 28 | 28 | 70,000 | 0 | 0 |
| fin/consumer_industrials | EvieKF | 0.03162 | 300 | 63 | 504 | 1,260,000 | 1 | 0 |
| fin/consumer_industrials | AdamW | 0.03162 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/consumer_industrials | AdaBelief | 0.03162 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/consumer_industrials | SGD | 0.9487 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/consumer_industrials | Sophia | 0.01 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/consumer_industrials | Muon | 0.006325 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/consumer_industrials | Lion | 0.0003162 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/consumer_industrials | Shampoo | 0.09999 | 0.95 | 36 | 288 | 720,000 | 1 | 0 |
| fin/finance | EvieKF | 0.01 | 9000 | 63 | 504 | 1,260,000 | 0 | 1 |
| fin/finance | AdamW | 0.01 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/finance | AdaBelief | 0.003162 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/finance | SGD | 0.9487 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/finance | Sophia | 0.001 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/finance | Muon | 0.0002 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/finance | Lion | 0.003162 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/finance | Shampoo | 0.003162 | 0.99 | 28 | 224 | 560,000 | 0 | 0 |
| fin/healthcare | EvieKF | 0.03162 | 1000 | 63 | 504 | 1,260,000 | 1 | 0 |
| fin/healthcare | AdamW | 0.003162 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/healthcare | AdaBelief | 0.003162 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/healthcare | SGD | 0.9487 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/healthcare | Sophia | 0.3162 | -- | 11 | 88 | 220,000 | 2 | 0 |
| fin/healthcare | Muon | 0.002 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/healthcare | Lion | 0.001 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/healthcare | Shampoo | 0.03162 | 0.99999 | 90 | 720 | 1,800,000 | 1 | 3 |
| fin/tech | EvieKF | 0.09999 | 10 | 63 | 504 | 1,260,000 | 1 | 0 |
| fin/tech | AdamW | 0.01 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/tech | AdaBelief | 0.01 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/tech | SGD | 0.9487 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/tech | Sophia | 0.003162 | -- | 7 | 56 | 140,000 | 0 | 0 |
| fin/tech | Muon | 0.6325 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/tech | Lion | 0.003162 | -- | 9 | 72 | 180,000 | 1 | 0 |
| fin/tech | Shampoo | 0.3162 | 0.95 | 44 | 352 | 880,000 | 2 | 0 |

## Table 7 -- sensitivity to the tuned second axis

LR-search score (best validation value, 1 seed, 25% of the step budget) at the frozen learning rate, for each value of the knob. A flat response means the knob does not matter on that task. Same units and scaling as Table 1. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Task | Optimizer | knob | val score |  |
|---|---|---|---|---|
| vis/cifar10 | EvieKF | 3 | 77.08 |  |
| vis/cifar10 | EvieKF | 10 | 78.28 |  |
| vis/cifar10 | EvieKF | 30 | 80.06 |  |
| vis/cifar10 | EvieKF | 100 | 79.60 |  |
| vis/cifar10 | EvieKF | 300 | 79.30 |  |
| vis/cifar10 | EvieKF | 1000 | 80.28 |  |
| vis/cifar10 | EvieKF | 3000 | 81.64 |  |
| vis/cifar10 | EvieKF | 9000 | 80.70 |  |
| vis/cifar10 | EvieKF | 27000 | 82.28 | frozen |
| vis/cifar10 | EvieKF | 81000 | 80.74 |  |
| vis/cifar10 | EvieKF | 243000 | 81.28 |  |
| vis/cifar10 | EvieKF | spread |  | 6.75% over 81000x |
| vis/cifar10 | Shampoo | 0.9 | 81.44 |  |
| vis/cifar10 | Shampoo | 0.95 | 82.42 | frozen |
| vis/cifar10 | Shampoo | 0.99 | 82.00 |  |
| vis/cifar10 | Shampoo | 0.999 | 81.62 |  |
| vis/cifar10 | Shampoo | spread |  | 1.20% over 1x |
| vis/cifar100 | EvieKF | 3 | 43.00 |  |
| vis/cifar100 | EvieKF | 10 | 44.18 |  |
| vis/cifar100 | EvieKF | 30 | 45.54 |  |
| vis/cifar100 | EvieKF | 100 | 47.82 |  |
| vis/cifar100 | EvieKF | 300 | 49.20 |  |
| vis/cifar100 | EvieKF | 1000 | 50.76 | frozen |
| vis/cifar100 | EvieKF | 3000 | 49.96 |  |
| vis/cifar100 | EvieKF | spread |  | 18.05% over 1000x |
| vis/cifar100 | Shampoo | 0.8 | 50.38 |  |
| vis/cifar100 | Shampoo | 0.85 | 50.50 | frozen |
| vis/cifar100 | Shampoo | 0.9 | 50.28 |  |
| vis/cifar100 | Shampoo | 0.95 | 50.08 |  |
| vis/cifar100 | Shampoo | 0.99 | 49.38 |  |
| vis/cifar100 | Shampoo | 0.999 | 49.54 |  |
| vis/cifar100 | Shampoo | spread |  | 2.27% over 1x |
| vis/stl10 | EvieKF | 0.3333 | 48.80 |  |
| vis/stl10 | EvieKF | 1 | 49.80 |  |
| vis/stl10 | EvieKF | 3 | 54.20 | frozen |
| vis/stl10 | EvieKF | 10 | 52.40 |  |
| vis/stl10 | EvieKF | 30 | 50.80 |  |
| vis/stl10 | EvieKF | 100 | 52.80 |  |
| vis/stl10 | EvieKF | 300 | 52.40 |  |
| vis/stl10 | EvieKF | 1000 | 51.60 |  |
| vis/stl10 | EvieKF | 3000 | 53.60 |  |
| vis/stl10 | EvieKF | spread |  | 11.07% over 9001x |
| vis/stl10 | Shampoo | 0.8 | 53.20 |  |
| vis/stl10 | Shampoo | 0.85 | 52.60 |  |
| vis/stl10 | Shampoo | 0.9 | 54.80 | frozen |
| vis/stl10 | Shampoo | 0.95 | 51.20 |  |
| vis/stl10 | Shampoo | 0.99 | 51.40 |  |
| vis/stl10 | Shampoo | 0.999 | 52.80 |  |
| vis/stl10 | Shampoo | spread |  | 7.03% over 1x |
| vis/svhn | EvieKF | 3 | 93.57 |  |
| vis/svhn | EvieKF | 10 | 93.83 | frozen |
| vis/svhn | EvieKF | 30 | 93.45 |  |
| vis/svhn | EvieKF | 100 | 93.42 |  |
| vis/svhn | EvieKF | 300 | 93.17 |  |
| vis/svhn | EvieKF | 1000 | 93.37 |  |
| vis/svhn | EvieKF | 3000 | 93.24 |  |
| vis/svhn | EvieKF | spread |  | 0.70% over 1000x |
| vis/svhn | Shampoo | 0.6 | 93.82 |  |
| vis/svhn | Shampoo | 0.65 | 93.86 |  |
| vis/svhn | Shampoo | 0.7 | 93.87 | frozen |
| vis/svhn | Shampoo | 0.75 | 93.79 |  |
| vis/svhn | Shampoo | 0.8 | 93.87 |  |
| vis/svhn | Shampoo | 0.85 | 93.80 |  |
| vis/svhn | Shampoo | 0.9 | 93.67 |  |
| vis/svhn | Shampoo | 0.95 | 93.76 |  |
| vis/svhn | Shampoo | 0.99 | 93.78 |  |
| vis/svhn | Shampoo | 0.999 | 93.80 |  |
| vis/svhn | Shampoo | spread |  | 0.22% over 2x |
| lang/enwik8 | EvieKF | 3 | 4.39 |  |
| lang/enwik8 | EvieKF | 10 | 4.31 |  |
| lang/enwik8 | EvieKF | 30 | 4.21 |  |
| lang/enwik8 | EvieKF | 100 | 4.14 |  |
| lang/enwik8 | EvieKF | 300 | 4.03 |  |
| lang/enwik8 | EvieKF | 1000 | 3.96 |  |
| lang/enwik8 | EvieKF | 3000 | 3.86 |  |
| lang/enwik8 | EvieKF | 9000 | 3.75 |  |
| lang/enwik8 | EvieKF | 27000 | 3.71 | frozen |
| lang/enwik8 | EvieKF | 81000 | 3.70 |  |
| lang/enwik8 | EvieKF | 243000 | 3.70 |  |
| lang/enwik8 | EvieKF | spread |  | 18.92% over 81000x |
| lang/enwik8 | Shampoo | 0.7 | 3.92 |  |
| lang/enwik8 | Shampoo | 0.75 | 3.79 | frozen |
| lang/enwik8 | Shampoo | 0.8 | 3.89 |  |
| lang/enwik8 | Shampoo | 0.85 | 3.97 |  |
| lang/enwik8 | Shampoo | 0.9 | 3.93 |  |
| lang/enwik8 | Shampoo | 0.95 | 3.96 |  |
| lang/enwik8 | Shampoo | 0.99 | 3.95 |  |
| lang/enwik8 | Shampoo | 0.999 | 4.85 |  |
| lang/enwik8 | Shampoo | spread |  | 27.84% over 1x |
| lang/ptb | EvieKF | 3 | 201.29 |  |
| lang/ptb | EvieKF | 10 | 186.28 | frozen |
| lang/ptb | EvieKF | 30 | 205.72 |  |
| lang/ptb | EvieKF | 100 | 198.82 |  |
| lang/ptb | EvieKF | 300 | 202.92 |  |
| lang/ptb | EvieKF | 1000 | 207.85 |  |
| lang/ptb | EvieKF | 3000 | 209.75 |  |
| lang/ptb | EvieKF | spread |  | 12.60% over 1000x |
| lang/ptb | Shampoo | 0.9 | 185.53 |  |
| lang/ptb | Shampoo | 0.95 | 184.92 |  |
| lang/ptb | Shampoo | 0.99 | 184.61 | frozen |
| lang/ptb | Shampoo | 0.999 | 186.85 |  |
| lang/ptb | Shampoo | spread |  | 1.21% over 1x |
| lang/wikitext103 | EvieKF | 3 | 77.49 |  |
| lang/wikitext103 | EvieKF | 10 | 78.72 |  |
| lang/wikitext103 | EvieKF | 30 | 79.11 |  |
| lang/wikitext103 | EvieKF | 100 | 74.50 |  |
| lang/wikitext103 | EvieKF | 300 | 74.37 |  |
| lang/wikitext103 | EvieKF | 1000 | 71.11 |  |
| lang/wikitext103 | EvieKF | 3000 | 66.28 |  |
| lang/wikitext103 | EvieKF | 9000 | 63.30 |  |
| lang/wikitext103 | EvieKF | 27000 | 61.34 |  |
| lang/wikitext103 | EvieKF | 81000 | 59.86 |  |
| lang/wikitext103 | EvieKF | 243000 | 59.18 | frozen |
| lang/wikitext103 | EvieKF | spread |  | 33.69% over 81000x |
| lang/wikitext103 | Shampoo | 0.8 | 62.94 | frozen |
| lang/wikitext103 | Shampoo | 0.85 | 63.14 |  |
| lang/wikitext103 | Shampoo | 0.9 | 63.01 |  |
| lang/wikitext103 | Shampoo | 0.95 | 63.46 |  |
| lang/wikitext103 | Shampoo | 0.99 | 65.93 |  |
| lang/wikitext103 | Shampoo | 0.999 | 71.50 |  |
| lang/wikitext103 | Shampoo | spread |  | 13.61% over 1x |
| lang/wikitext2 | EvieKF | 3 | 100.50 |  |
| lang/wikitext2 | EvieKF | 10 | 101.69 |  |
| lang/wikitext2 | EvieKF | 30 | 98.39 |  |
| lang/wikitext2 | EvieKF | 100 | 97.69 |  |
| lang/wikitext2 | EvieKF | 300 | 97.60 | frozen |
| lang/wikitext2 | EvieKF | 1000 | 98.34 |  |
| lang/wikitext2 | EvieKF | 3000 | 99.66 |  |
| lang/wikitext2 | EvieKF | spread |  | 4.19% over 1000x |
| lang/wikitext2 | Shampoo | 0.9 | 122.47 |  |
| lang/wikitext2 | Shampoo | 0.95 | 121.90 | frozen |
| lang/wikitext2 | Shampoo | 0.99 | 125.46 |  |
| lang/wikitext2 | Shampoo | 0.999 | 132.92 |  |
| lang/wikitext2 | Shampoo | spread |  | 9.04% over 1x |
| fin/consumer_industrials | EvieKF | 3 | 3.6213 |  |
| fin/consumer_industrials | EvieKF | 10 | 3.6480 |  |
| fin/consumer_industrials | EvieKF | 30 | 3.6003 |  |
| fin/consumer_industrials | EvieKF | 100 | 3.6108 |  |
| fin/consumer_industrials | EvieKF | 300 | 3.5750 | frozen |
| fin/consumer_industrials | EvieKF | 1000 | 3.6149 |  |
| fin/consumer_industrials | EvieKF | 3000 | 3.6329 |  |
| fin/consumer_industrials | EvieKF | spread |  | 2.04% over 1000x |
| fin/consumer_industrials | Shampoo | 0.9 | 3.6348 |  |
| fin/consumer_industrials | Shampoo | 0.95 | 3.6040 | frozen |
| fin/consumer_industrials | Shampoo | 0.99 | 3.6248 |  |
| fin/consumer_industrials | Shampoo | 0.999 | 3.6173 |  |
| fin/consumer_industrials | Shampoo | spread |  | 0.85% over 1x |
| fin/finance | EvieKF | 3 | 3.6488 |  |
| fin/finance | EvieKF | 10 | 3.6415 |  |
| fin/finance | EvieKF | 30 | 3.6468 |  |
| fin/finance | EvieKF | 100 | 3.6419 |  |
| fin/finance | EvieKF | 300 | 3.6355 |  |
| fin/finance | EvieKF | 1000 | 3.6358 |  |
| fin/finance | EvieKF | 3000 | 3.6276 |  |
| fin/finance | EvieKF | 9000 | 3.6253 | frozen |
| fin/finance | EvieKF | 27000 | 3.6331 |  |
| fin/finance | EvieKF | spread |  | 0.65% over 9000x |
| fin/finance | Shampoo | 0.9 | 3.6378 |  |
| fin/finance | Shampoo | 0.95 | 3.6333 |  |
| fin/finance | Shampoo | 0.99 | 3.6294 | frozen |
| fin/finance | Shampoo | 0.999 | 3.6343 |  |
| fin/finance | Shampoo | spread |  | 0.23% over 1x |
| fin/healthcare | EvieKF | 3 | 2.4112 |  |
| fin/healthcare | EvieKF | 10 | 2.4283 |  |
| fin/healthcare | EvieKF | 30 | 2.4098 |  |
| fin/healthcare | EvieKF | 100 | 2.4286 |  |
| fin/healthcare | EvieKF | 300 | 2.4039 |  |
| fin/healthcare | EvieKF | 1000 | 2.3772 | frozen |
| fin/healthcare | EvieKF | 3000 | 2.4010 |  |
| fin/healthcare | EvieKF | spread |  | 2.16% over 1000x |
| fin/healthcare | Shampoo | 0.9 | 2.3929 |  |
| fin/healthcare | Shampoo | 0.95 | 2.3934 |  |
| fin/healthcare | Shampoo | 0.99 | 2.3994 |  |
| fin/healthcare | Shampoo | 0.999 | 2.3693 |  |
| fin/healthcare | Shampoo | 0.999667 | 2.3750 |  |
| fin/healthcare | Shampoo | 0.9999 | 2.3664 |  |
| fin/healthcare | Shampoo | 0.999967 | 2.3639 |  |
| fin/healthcare | Shampoo | 0.99999 | 2.3617 | frozen |
| fin/healthcare | Shampoo | 0.999997 | 2.3638 |  |
| fin/healthcare | Shampoo | 0.999999 | 2.3650 |  |
| fin/healthcare | Shampoo | spread |  | 1.60% over 1x |
| fin/tech | EvieKF | 3 | 5.3854 |  |
| fin/tech | EvieKF | 10 | 5.3715 | frozen |
| fin/tech | EvieKF | 30 | 5.3978 |  |
| fin/tech | EvieKF | 100 | 5.3803 |  |
| fin/tech | EvieKF | 300 | 5.3774 |  |
| fin/tech | EvieKF | 1000 | 5.3720 |  |
| fin/tech | EvieKF | 3000 | 5.4693 |  |
| fin/tech | EvieKF | spread |  | 1.82% over 1000x |
| fin/tech | Shampoo | 0.9 | 5.3845 |  |
| fin/tech | Shampoo | 0.95 | 5.3660 | frozen |
| fin/tech | Shampoo | 0.99 | 5.3953 |  |
| fin/tech | Shampoo | 0.999 | 5.3832 |  |
| fin/tech | Shampoo | spread |  | 0.55% over 1x |

## Table 8 -- compute per training cell

Mean wall-clock seconds and peak GPU memory per (optimizer, unit, seed) cell of 10,000 steps, on the GPU listed in SETUP.md. The step budget is fixed, not wall-clock matched. Cells recovered from logs carry no wall-clock, so 'n' can be below the full count. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Task | Optimizer | n | s / cell | vs AdamW | peak MB |
|---|---|---|---|---|---|
| vis/cifar10 | EvieKF | 5 | 1764.5 | 1.59x | 851 |
| vis/cifar10 | AdamW | 5 | 1108.0 | 1.00x | 780 |
| vis/cifar10 | AdaBelief | 5 | 1154.3 | 1.04x | 780 |
| vis/cifar10 | SGD | 5 | 1151.0 | 1.04x | 737 |
| vis/cifar10 | Sophia | 5 | 4764.3 | 4.30x | 2349 |
| vis/cifar10 | Muon | 5 | 1509.3 | 1.36x | 737 |
| vis/cifar10 | Lion | 5 | 1132.1 | 1.02x | 737 |
| vis/cifar10 | Shampoo | 5 | 1149.7 | 1.04x | 782 |
| vis/cifar100 | EvieKF | 5 | 1808.2 | 1.63x | 852 |
| vis/cifar100 | AdamW | 5 | 1108.8 | 1.00x | 781 |
| vis/cifar100 | AdaBelief | 5 | 1151.8 | 1.04x | 781 |
| vis/cifar100 | SGD | 5 | 1099.4 | 0.99x | 738 |
| vis/cifar100 | Sophia | 5 | 5245.2 | 4.73x | 2343 |
| vis/cifar100 | Muon | 5 | 1503.9 | 1.36x | 738 |
| vis/cifar100 | Lion | 5 | 1105.2 | 1.00x | 738 |
| vis/cifar100 | Shampoo | 5 | 1171.4 | 1.06x | 783 |
| vis/stl10 | EvieKF | 5 | 1817.1 | 1.60x | 851 |
| vis/stl10 | AdamW | 5 | 1135.6 | 1.00x | 780 |
| vis/stl10 | AdaBelief | 5 | 1092.3 | 0.96x | 780 |
| vis/stl10 | SGD | 5 | 1133.3 | 1.00x | 737 |
| vis/stl10 | Sophia | 5 | 4611.4 | 4.06x | 2349 |
| vis/stl10 | Muon | 5 | 1422.7 | 1.25x | 737 |
| vis/stl10 | Lion | 5 | 1129.9 | 0.99x | 737 |
| vis/stl10 | Shampoo | 5 | 1100.7 | 0.97x | 782 |
| vis/svhn | EvieKF | 5 | 1902.7 | 1.65x | 851 |
| vis/svhn | AdamW | 5 | 1155.2 | 1.00x | 780 |
| vis/svhn | AdaBelief | 5 | 1205.0 | 1.04x | 780 |
| vis/svhn | SGD | 5 | 1083.3 | 0.94x | 737 |
| vis/svhn | Sophia | 5 | 4753.2 | 4.11x | 2349 |
| vis/svhn | Muon | 5 | 1583.9 | 1.37x | 737 |
| vis/svhn | Lion | 5 | 1075.1 | 0.93x | 737 |
| vis/svhn | Shampoo | 5 | 1235.2 | 1.07x | 782 |
| lang/enwik8 | EvieKF | 5 | 3562.9 | 1.64x | 923 |
| lang/enwik8 | AdamW | 5 | 2171.9 | 1.00x | 1378 |
| lang/enwik8 | AdaBelief | 5 | 2199.5 | 1.01x | 1378 |
| lang/enwik8 | SGD | 5 | 2136.3 | 0.98x | 1359 |
| lang/enwik8 | Sophia | 5 | 5325.3 | 2.45x | 4879 |
| lang/enwik8 | Muon | 5 | 2332.1 | 1.07x | 1360 |
| lang/enwik8 | Lion | 5 | 2145.1 | 0.99x | 1359 |
| lang/enwik8 | Shampoo | 5 | 3039.9 | 1.40x | 1518 |
| lang/ptb | EvieKF | 5 | 3173.3 | 1.72x | 1592 |
| lang/ptb | AdamW | 5 | 1846.8 | 1.00x | 2264 |
| lang/ptb | AdaBelief | 5 | 2197.6 | 1.19x | 2264 |
| lang/ptb | SGD | 5 | 1835.4 | 0.99x | 2236 |
| lang/ptb | Sophia | 5 | 6079.0 | 3.29x | 7091 |
| lang/ptb | Muon | 5 | 2332.1 | 1.26x | 2246 |
| lang/ptb | Lion | 5 | 1856.0 | 1.00x | 2236 |
| lang/ptb | Shampoo | 5 | 3101.8 | 1.68x | 2403 |
| lang/wikitext103 | EvieKF | 5 | 3674.0 | 1.50x | 2370 |
| lang/wikitext103 | AdamW | 5 | 2447.0 | 1.00x | 2856 |
| lang/wikitext103 | AdaBelief | 5 | 2458.6 | 1.00x | 2856 |
| lang/wikitext103 | SGD | 5 | 2131.3 | 0.87x | 2822 |
| lang/wikitext103 | Sophia | 5 | 6798.6 | 2.78x | 8642 |
| lang/wikitext103 | Muon | 5 | 2460.9 | 1.01x | 2838 |
| lang/wikitext103 | Lion | 5 | 2142.1 | 0.88x | 2822 |
| lang/wikitext103 | Shampoo | 5 | 3147.8 | 1.29x | 2995 |
| lang/wikitext2 | EvieKF | 5 | 3876.2 | 1.57x | 2377 |
| lang/wikitext2 | AdamW | 5 | 2464.7 | 1.00x | 2856 |
| lang/wikitext2 | AdaBelief | 5 | 2473.1 | 1.00x | 2856 |
| lang/wikitext2 | SGD | 5 | 2169.3 | 0.88x | 2822 |
| lang/wikitext2 | Sophia | 5 | 6902.0 | 2.80x | 8656 |
| lang/wikitext2 | Muon | 5 | 2535.6 | 1.03x | 2838 |
| lang/wikitext2 | Lion | 5 | 2158.5 | 0.88x | 2822 |
| lang/wikitext2 | Shampoo | 5 | 3253.2 | 1.32x | 2995 |
| fin/consumer_industrials | EvieKF | 250 | 81.7 | 4.72x | 19 |
| fin/consumer_industrials | AdamW | 250 | 17.3 | 1.00x | 17 |
| fin/consumer_industrials | AdaBelief | 250 | 22.3 | 1.29x | 17 |
| fin/consumer_industrials | SGD | 250 | 18.1 | 1.04x | 17 |
| fin/consumer_industrials | Sophia | 250 | 43.6 | 2.52x | 18 |
| fin/consumer_industrials | Muon | 250 | 40.0 | 2.31x | 17 |
| fin/consumer_industrials | Lion | 250 | 19.2 | 1.11x | 17 |
| fin/consumer_industrials | Shampoo | 250 | 39.1 | 2.26x | 18 |
| fin/finance | EvieKF | 245 | 93.4 | 4.73x | 19 |
| fin/finance | AdamW | 245 | 19.8 | 1.00x | 18 |
| fin/finance | AdaBelief | 245 | 25.4 | 1.29x | 18 |
| fin/finance | SGD | 245 | 17.5 | 0.89x | 17 |
| fin/finance | Sophia | 245 | 45.7 | 2.31x | 18 |
| fin/finance | Muon | 245 | 43.6 | 2.21x | 17 |
| fin/finance | Lion | 245 | 20.3 | 1.03x | 17 |
| fin/finance | Shampoo | 245 | 38.5 | 1.95x | 18 |
| fin/healthcare | EvieKF | 245 | 85.1 | 4.75x | 19 |
| fin/healthcare | AdamW | 245 | 17.9 | 1.00x | 17 |
| fin/healthcare | AdaBelief | 245 | 23.3 | 1.30x | 17 |
| fin/healthcare | SGD | 245 | 18.4 | 1.03x | 17 |
| fin/healthcare | Sophia | 245 | 43.3 | 2.42x | 18 |
| fin/healthcare | Muon | 245 | 41.4 | 2.31x | 17 |
| fin/healthcare | Lion | 245 | 19.4 | 1.08x | 17 |
| fin/healthcare | Shampoo | 245 | 37.3 | 2.08x | 18 |
| fin/tech | EvieKF | 135 | 86.9 | 4.87x | 19 |
| fin/tech | AdamW | 245 | 17.9 | 1.00x | 17 |
| fin/tech | AdaBelief | 245 | 23.0 | 1.29x | 17 |
| fin/tech | SGD | 120 | 17.9 | 1.00x | 17 |
| fin/tech | Shampoo | 70 | 37.6 | 2.11x | 18 |

## Table 9 -- divergence and complete-case units

Cells flagged diverged (non-finite loss, outside the task's absolute sane range, or beyond the peer-median factor), and the units on which they occurred. A unit on which any arm failed is dropped for every arm, so no arm is scored only on the units it survived. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Task | Optimizer | diverged | units |
|---|---|---|---|
| vis/cifar10 | EvieKF | 0/5 | -- |
| vis/cifar10 | AdamW | 0/5 | -- |
| vis/cifar10 | AdaBelief | 0/5 | -- |
| vis/cifar10 | SGD | 0/5 | -- |
| vis/cifar10 | Sophia | 0/5 | -- |
| vis/cifar10 | Muon | 0/5 | -- |
| vis/cifar10 | Lion | 0/5 | -- |
| vis/cifar10 | Shampoo | 0/5 | -- |
| vis/cifar10 | complete-case | 1/1 units | none dropped |
| vis/cifar100 | EvieKF | 0/5 | -- |
| vis/cifar100 | AdamW | 0/5 | -- |
| vis/cifar100 | AdaBelief | 0/5 | -- |
| vis/cifar100 | SGD | 0/5 | -- |
| vis/cifar100 | Sophia | 0/5 | -- |
| vis/cifar100 | Muon | 0/5 | -- |
| vis/cifar100 | Lion | 0/5 | -- |
| vis/cifar100 | Shampoo | 0/5 | -- |
| vis/cifar100 | complete-case | 1/1 units | none dropped |
| vis/stl10 | EvieKF | 0/5 | -- |
| vis/stl10 | AdamW | 0/5 | -- |
| vis/stl10 | AdaBelief | 0/5 | -- |
| vis/stl10 | SGD | 0/5 | -- |
| vis/stl10 | Sophia | 0/5 | -- |
| vis/stl10 | Muon | 0/5 | -- |
| vis/stl10 | Lion | 0/5 | -- |
| vis/stl10 | Shampoo | 0/5 | -- |
| vis/stl10 | complete-case | 1/1 units | none dropped |
| vis/svhn | EvieKF | 0/5 | -- |
| vis/svhn | AdamW | 0/5 | -- |
| vis/svhn | AdaBelief | 0/5 | -- |
| vis/svhn | SGD | 0/5 | -- |
| vis/svhn | Sophia | 0/5 | -- |
| vis/svhn | Muon | 0/5 | -- |
| vis/svhn | Lion | 0/5 | -- |
| vis/svhn | Shampoo | 0/5 | -- |
| vis/svhn | complete-case | 1/1 units | none dropped |
| lang/enwik8 | EvieKF | 0/5 | -- |
| lang/enwik8 | AdamW | 0/5 | -- |
| lang/enwik8 | AdaBelief | 0/5 | -- |
| lang/enwik8 | SGD | 0/5 | -- |
| lang/enwik8 | Sophia | 0/5 | -- |
| lang/enwik8 | Muon | 0/5 | -- |
| lang/enwik8 | Lion | 0/5 | -- |
| lang/enwik8 | Shampoo | 0/5 | -- |
| lang/enwik8 | complete-case | 1/1 units | none dropped |
| lang/ptb | EvieKF | 0/5 | -- |
| lang/ptb | AdamW | 0/5 | -- |
| lang/ptb | AdaBelief | 0/5 | -- |
| lang/ptb | SGD | 0/5 | -- |
| lang/ptb | Sophia | 0/5 | -- |
| lang/ptb | Muon | 0/5 | -- |
| lang/ptb | Lion | 0/5 | -- |
| lang/ptb | Shampoo | 0/5 | -- |
| lang/ptb | complete-case | 1/1 units | none dropped |
| lang/wikitext103 | EvieKF | 0/5 | -- |
| lang/wikitext103 | AdamW | 0/5 | -- |
| lang/wikitext103 | AdaBelief | 0/5 | -- |
| lang/wikitext103 | SGD | 0/5 | -- |
| lang/wikitext103 | Sophia | 0/5 | -- |
| lang/wikitext103 | Muon | 0/5 | -- |
| lang/wikitext103 | Lion | 0/5 | -- |
| lang/wikitext103 | Shampoo | 0/5 | -- |
| lang/wikitext103 | complete-case | 1/1 units | none dropped |
| lang/wikitext2 | EvieKF | 0/5 | -- |
| lang/wikitext2 | AdamW | 0/5 | -- |
| lang/wikitext2 | AdaBelief | 0/5 | -- |
| lang/wikitext2 | SGD | 0/5 | -- |
| lang/wikitext2 | Sophia | 0/5 | -- |
| lang/wikitext2 | Muon | 0/5 | -- |
| lang/wikitext2 | Lion | 0/5 | -- |
| lang/wikitext2 | Shampoo | 0/5 | -- |
| lang/wikitext2 | complete-case | 1/1 units | none dropped |
| fin/consumer_industrials | EvieKF | 0/250 | -- |
| fin/consumer_industrials | AdamW | 0/250 | -- |
| fin/consumer_industrials | AdaBelief | 0/250 | -- |
| fin/consumer_industrials | SGD | 3/250 | CMI, NKE |
| fin/consumer_industrials | Sophia | 0/250 | -- |
| fin/consumer_industrials | Muon | 0/250 | -- |
| fin/consumer_industrials | Lion | 0/250 | -- |
| fin/consumer_industrials | Shampoo | 3/250 | CARR, GWW, PH |
| fin/consumer_industrials | complete-case | 45/50 units | CARR, CMI, GWW, NKE, PH |
| fin/finance | EvieKF | 0/245 | -- |
| fin/finance | AdamW | 0/245 | -- |
| fin/finance | AdaBelief | 0/245 | -- |
| fin/finance | SGD | 4/245 | MKL, NDAQ, PGR, SCHW |
| fin/finance | Sophia | 2/245 | ACGL, PGR |
| fin/finance | Muon | 32/245 | ACGL, AFL, ALL, AMP, AXP, BRO, CB, CBOE, GL, GS, HIG, JPM, L, MET, PGR, RJF, TRV, WRB |
| fin/finance | Lion | 17/245 | ACGL, AIG, AMP, AXP, BLK, CB, CBOE, CME, GL, HIG, JPM, PNC, RF, TRV |
| fin/finance | Shampoo | 11/245 | ACGL, AFL, AXP, GL, L, PGR |
| fin/finance | complete-case | 23/49 units | ACGL, AFL, AIG, ALL, AMP, AXP, BLK, BRO, CB, CBOE, CME, GL, GS, HIG, JPM, L, MET, MKL, NDAQ, PGR, PNC, RF, RJF, SCHW, TRV, WRB |
| fin/healthcare | EvieKF | 0/245 | -- |
| fin/healthcare | AdamW | 0/245 | -- |
| fin/healthcare | AdaBelief | 0/245 | -- |
| fin/healthcare | SGD | 4/245 | CAH, ELV, EW, HCA |
| fin/healthcare | Sophia | 14/245 | AMGN, CAH, CI, DVA, EW, HCA, LLY, MDT, REGN, SYK, TMO, VRTX |
| fin/healthcare | Muon | 2/245 | LLY, MCK |
| fin/healthcare | Lion | 5/245 | BSX, JNJ, LLY, MCK, VRTX |
| fin/healthcare | Shampoo | 0/245 | -- |
| fin/healthcare | complete-case | 33/49 units | AMGN, BSX, CAH, CI, DVA, ELV, EW, HCA, JNJ, LLY, MCK, MDT, REGN, SYK, TMO, VRTX |
| fin/tech | EvieKF | 2/245 | IBM |
| fin/tech | AdamW | 0/245 | -- |
| fin/tech | AdaBelief | 0/245 | -- |
| fin/tech | SGD | 2/245 | CRM, FICO |
| fin/tech | Sophia | 2/245 | NVDA |
| fin/tech | Muon | 0/245 | -- |
| fin/tech | Lion | 8/245 | AAPL, APH, CDNS, CRM, GOOGL, KLAC, MCHP, TXN |
| fin/tech | Shampoo | 7/245 | APH, FICO, HPE, IBM |
| fin/tech | complete-case | 37/49 units | AAPL, APH, CDNS, CRM, FICO, GOOGL, HPE, IBM, KLAC, MCHP, NVDA, TXN |

## Table 9b -- sensitivity: failures counted as losses

Primary aggregation (complete-case: a unit any arm failed on is dropped for all arms) against a sensitivity that keeps every unit and scores each diverged cell as the worst finite value any arm reached on that unit. Under complete-case an arm's own failures do not enter its task number; here they do. Same units and scaling as Table 1. Sensitivity analysis only. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Task | Optimizer | diverged cells | complete-case | failures as losses | rank (CC) | rank (failures) |
|---|---|---|---|---|---|---|
| vis/cifar10 | EvieKF | 0 | 81.94 | 81.94 | 2 | 2 |
| vis/cifar10 | AdamW | 0 | 81.55 | 81.55 | 5 | 5 |
| vis/cifar10 | AdaBelief | 0 | 81.00 | 81.00 | 8 | 8 |
| vis/cifar10 | SGD | 0 | 81.71 | 81.71 | 4 | 4 |
| vis/cifar10 | Sophia | 0 | 81.52 | 81.52 | 6 | 6 |
| vis/cifar10 | Muon | 0 | 85.86 | 85.86 | 1 | 1 |
| vis/cifar10 | Lion | 0 | 81.28 | 81.28 | 7 | 7 |
| vis/cifar10 | Shampoo | 0 | 81.90 | 81.90 | 3 | 3 |
| vis/cifar100 | EvieKF | 0 | 52.28 | 52.28 | 3 | 3 |
| vis/cifar100 | AdamW | 0 | 49.51 | 49.51 | 6 | 6 |
| vis/cifar100 | AdaBelief | 0 | 49.45 | 49.45 | 7 | 7 |
| vis/cifar100 | SGD | 0 | 54.27 | 54.27 | 1 | 1 |
| vis/cifar100 | Sophia | 0 | 47.39 | 47.39 | 8 | 8 |
| vis/cifar100 | Muon | 0 | 54.24 | 54.24 | 2 | 2 |
| vis/cifar100 | Lion | 0 | 49.59 | 49.59 | 5 | 5 |
| vis/cifar100 | Shampoo | 0 | 50.80 | 50.80 | 4 | 4 |
| vis/stl10 | EvieKF | 0 | 48.59 | 48.59 | 6 | 6 |
| vis/stl10 | AdamW | 0 | 51.29 | 51.29 | 1 | 1 |
| vis/stl10 | AdaBelief | 0 | 50.84 | 50.84 | 3 | 3 |
| vis/stl10 | SGD | 0 | 47.75 | 47.75 | 7 | 7 |
| vis/stl10 | Sophia | 0 | 47.48 | 47.48 | 8 | 8 |
| vis/stl10 | Muon | 0 | 48.85 | 48.85 | 5 | 5 |
| vis/stl10 | Lion | 0 | 49.96 | 49.96 | 4 | 4 |
| vis/stl10 | Shampoo | 0 | 51.25 | 51.25 | 2 | 2 |
| vis/svhn | EvieKF | 0 | 94.63 | 94.63 | 7 | 7 |
| vis/svhn | AdamW | 0 | 94.66 | 94.66 | 6 | 6 |
| vis/svhn | AdaBelief | 0 | 94.75 | 94.75 | 4 | 4 |
| vis/svhn | SGD | 0 | 94.73 | 94.73 | 5 | 5 |
| vis/svhn | Sophia | 0 | 94.89 | 94.89 | 2 | 2 |
| vis/svhn | Muon | 0 | 95.52 | 95.52 | 1 | 1 |
| vis/svhn | Lion | 0 | 94.55 | 94.55 | 8 | 8 |
| vis/svhn | Shampoo | 0 | 94.83 | 94.83 | 3 | 3 |
| lang/enwik8 | EvieKF | 0 | 3.04 | 3.04 | 3 | 3 |
| lang/enwik8 | AdamW | 0 | 3.18 | 3.18 | 6 | 6 |
| lang/enwik8 | AdaBelief | 0 | 3.15 | 3.15 | 5 | 5 |
| lang/enwik8 | SGD | 0 | 3.60 | 3.60 | 8 | 8 |
| lang/enwik8 | Sophia | 0 | 3.34 | 3.34 | 7 | 7 |
| lang/enwik8 | Muon | 0 | 2.97 | 2.97 | 1 | 1 |
| lang/enwik8 | Lion | 0 | 3.11 | 3.11 | 4 | 4 |
| lang/enwik8 | Shampoo | 0 | 2.97 | 2.97 | 2 | 2 |
| lang/ptb | EvieKF | 0 | 192.17 | 192.17 | 6 | 6 |
| lang/ptb | AdamW | 0 | 179.35 | 179.35 | 2 | 2 |
| lang/ptb | AdaBelief | 0 | 179.77 | 179.77 | 3 | 3 |
| lang/ptb | SGD | 0 | 180.35 | 180.35 | 4 | 4 |
| lang/ptb | Sophia | 0 | 201.49 | 201.49 | 7 | 7 |
| lang/ptb | Muon | 0 | 187.01 | 187.01 | 5 | 5 |
| lang/ptb | Lion | 0 | 201.94 | 201.94 | 8 | 8 |
| lang/ptb | Shampoo | 0 | 172.94 | 172.94 | 1 | 1 |
| lang/wikitext103 | EvieKF | 0 | 35.41 | 35.41 | 2 | 2 |
| lang/wikitext103 | AdamW | 0 | 39.99 | 39.99 | 5 | 5 |
| lang/wikitext103 | AdaBelief | 0 | 41.17 | 41.17 | 6 | 6 |
| lang/wikitext103 | SGD | 0 | 59.75 | 59.75 | 8 | 8 |
| lang/wikitext103 | Sophia | 0 | 39.51 | 39.51 | 4 | 4 |
| lang/wikitext103 | Muon | 0 | 33.49 | 33.49 | 1 | 1 |
| lang/wikitext103 | Lion | 0 | 47.31 | 47.31 | 7 | 7 |
| lang/wikitext103 | Shampoo | 0 | 36.25 | 36.25 | 3 | 3 |
| lang/wikitext2 | EvieKF | 0 | 100.46 | 100.46 | 3 | 3 |
| lang/wikitext2 | AdamW | 0 | 105.70 | 105.70 | 5 | 5 |
| lang/wikitext2 | AdaBelief | 0 | 97.15 | 97.15 | 2 | 2 |
| lang/wikitext2 | SGD | 0 | 103.80 | 103.80 | 4 | 4 |
| lang/wikitext2 | Sophia | 0 | 123.75 | 123.75 | 7 | 7 |
| lang/wikitext2 | Muon | 0 | 83.62 | 83.62 | 1 | 1 |
| lang/wikitext2 | Lion | 0 | 106.65 | 106.65 | 6 | 6 |
| lang/wikitext2 | Shampoo | 0 | 123.91 | 123.91 | 8 | 8 |
| fin/consumer_industrials | EvieKF | 0 | 1.9687 | 2.1447 | 3 | 5 |
| fin/consumer_industrials | AdamW | 0 | 2.0169 | 2.0589 | 4 | 3 |
| fin/consumer_industrials | AdaBelief | 0 | 2.1041 | 2.1744 | 7 | 6 |
| fin/consumer_industrials | SGD | 3 | 1.8491 | 1.8973 | 1 | 1 |
| fin/consumer_industrials | Sophia | 0 | 2.0623 | 2.1370 | 6 | 4 |
| fin/consumer_industrials | Muon | 0 | 1.9124 | 1.9755 | 2 | 2 |
| fin/consumer_industrials | Lion | 0 | 2.1161 | 2.2449 | 8 | 8 |
| fin/consumer_industrials | Shampoo | 3 | 2.0574 | 2.2115 | 5 | 7 |
| fin/finance | EvieKF | 0 | 2.2669 | 2.0183 | 5 | 2 |
| fin/finance | AdamW | 0 | 2.2643 | 2.2044 | 4 | 4 |
| fin/finance | AdaBelief | 0 | 2.2583 | 2.4062 | 3 | 5 |
| fin/finance | SGD | 4 | 2.2278 | 1.8437 | 2 | 1 |
| fin/finance | Sophia | 2 | 2.6356 | 3.2484 | 6 | 6 |
| fin/finance | Muon | 32 | 3.7243 | 6.0995 | 8 | 8 |
| fin/finance | Lion | 17 | 2.2269 | 2.0545 | 1 | 3 |
| fin/finance | Shampoo | 11 | 3.3812 | 4.3552 | 7 | 7 |
| fin/healthcare | EvieKF | 0 | 2.9150 | 2.5137 | 2 | 1 |
| fin/healthcare | AdamW | 0 | 3.0969 | 3.0969 | 3 | 7 |
| fin/healthcare | AdaBelief | 0 | 3.1291 | 3.0952 | 6 | 6 |
| fin/healthcare | SGD | 4 | 2.9077 | 2.5923 | 1 | 2 |
| fin/healthcare | Sophia | 14 | 3.1868 | 2.7112 | 7 | 3 |
| fin/healthcare | Muon | 2 | 3.2588 | 3.2588 | 8 | 8 |
| fin/healthcare | Lion | 5 | 3.1281 | 2.7113 | 5 | 4 |
| fin/healthcare | Shampoo | 0 | 3.1132 | 2.9414 | 4 | 5 |
| fin/tech | EvieKF | 2 | 4.1497 | 3.9102 | 2 | 2 |
| fin/tech | AdamW | 0 | 4.3336 | 4.3336 | 6 | 6 |
| fin/tech | AdaBelief | 0 | 4.3745 | 4.4542 | 7 | 7 |
| fin/tech | SGD | 2 | 4.1537 | 4.1503 | 4 | 4 |
| fin/tech | Sophia | 2 | 4.5594 | 4.7083 | 8 | 8 |
| fin/tech | Muon | 0 | 4.1475 | 3.9095 | 1 | 1 |
| fin/tech | Lion | 8 | 4.1502 | 4.1465 | 3 | 3 |
| fin/tech | Shampoo | 7 | 4.2170 | 4.3123 | 5 | 5 |

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

## Table 10 -- EvieKF mechanism diagnostics

Mean [min, max] over cells. cos(update, Adam dir) = 1 means the operator left the AdamW direction unchanged. b = tr(Sigma_z)/||y||^2, the gradient noise relative to signal. Effective rank of the Kronecker noise spectrum. min gain = 1 means no shrinkage. Caveats: gains are recorded only in the Kronecker (2-D) branch, not the 1-D/diagonal branch; and the active-step counter resets when a cell resumes across a session break, so it is omitted here. Descriptive only. Covers 12 of 12 tasks: vision_cifar10, vision_cifar100, vision_stl10, vision_svhn, language_enwik8, language_ptb, language_wikitext103, language_wikitext2, finance_consumer_industrials, finance_finance, finance_healthcare, finance_tech.

| Task | cells | cos(update, Adam dir) | noise/signal b | noise eff. rank | min gain |
|---|---|---|---|---|---|
| vis/cifar10 | 5 | 0.468 [0.457, 0.477] | 15.7 [13.6, 17.5] | 3.5e+04 [3.16e+04, 3.88e+04] | 0.00553 [0.00487, 0.00656] |
| vis/cifar100 | 5 | 0.794 [0.779, 0.806] | 12.3 [11.3, 13.6] | 1.2e+05 [1.14e+05, 1.26e+05] | 0.00682 [0.00503, 0.00832] |
| vis/stl10 | 5 | 0.881 [0.852, 0.908] | 7.76 [6.93, 8.56] | 8.83e+03 [8.24e+03, 9.41e+03] | 0.761 [0.738, 0.789] |
| vis/svhn | 5 | 0.942 [0.94, 0.943] | 15.3 [13.1, 18] | 4.5e+04 [4.25e+04, 4.72e+04] | 0.277 [0.223, 0.301] |
| lang/enwik8 | 5 | 0.595 [0.591, 0.599] | 19.1 [17.9, 19.6] | 1.9e+03 [1.68e+03, 2.1e+03] | 0.00143 [0.00138, 0.00147] |
| lang/ptb | 5 | 0.993 [0.992, 0.993] | 9.49 [9.13, 9.73] | 3.46e+04 [3.34e+04, 3.54e+04] | 0.321 [0.276, 0.373] |
| lang/wikitext103 | 5 | 0.559 [0.551, 0.567] | 18.7 [18.5, 19] | 7.87e+03 [7.65e+03, 8e+03] | 0.000915 [0.000825, 0.000975] |
| lang/wikitext2 | 5 | 0.905 [0.902, 0.909] | 10.5 [10.1, 10.9] | 4.18e+04 [3.81e+04, 4.4e+04] | 0.0504 [0.0448, 0.0597] |
| fin/consumer_industrials | 250 | 0.798 [0.668, 0.883] | 38.7 [12.7, 92.9] | 5.53 [3.35, 10.9] | 0.601 [0.017, 0.871] |
| fin/finance | 245 | 0.596 [0.428, 0.753] | 46.9 [14.4, 172] | 3.27 [2.04, 8.15] | 0.0958 [0.00271, 0.25] |
| fin/healthcare | 245 | 0.757 [0.588, 0.854] | 46.3 [10.9, 304] | 5.37 [3.13, 9.79] | 0.363 [0.00836, 0.694] |
| fin/tech | 135 | 0.922 [0.869, 0.989] | 203 [15, 1.58e+03] | 4.09 [1.57, 11.1] | 0.98 [0.274, 1] |
