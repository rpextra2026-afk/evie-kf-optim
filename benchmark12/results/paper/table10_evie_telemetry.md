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
