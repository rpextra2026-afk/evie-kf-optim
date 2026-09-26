# Where the comments point

The code in this harness was written against a set of internal working
documents. Their names survive in comments — `CLAUDE.md sec 2`,
`claude_optim.md sec 4.1`, `DERIVATION.md sec 5.1`, and so on — because the
generated task files are byte-identical copies of the fragments that produced
every number in the paper, and rewriting those comments would mean regenerating
files that no longer match what actually ran.

The documents themselves are working notes and are not part of the release.
Everything they specify is in the paper. This table says where.

| comment reference | what it specified | where it is now |
|---|---|---|
| `CLAUDE.md` sec 1–2 | file convention: one self-contained script per task, one shared optimizer block | Appendix H.1; `_build/build.py` |
| `CLAUDE.md` sec 3 | the twelve tasks, models and datasets | Section 4, *Tasks and models*; Appendix H.3 |
| `CLAUDE.md` sec 4 | the baseline set and their frozen hyperparameters | Section 4, *Baselines*; Appendix H.3 |
| `CLAUDE.md` sec 5 | the protocol: 7-point grid, boundary rule, 5 seeds, one metric per domain, task as unit of analysis | Section 4, *Protocol* and *Hyperparameter tuning* |
| `CLAUDE.md` sec 6 | checkpointing, resume, divergence detection | Appendix H.3; `_build/frag_engine.py` |
| `CLAUDE.md` sec 9 | the batch-size and model-size sweep design | Appendix G.2 |
| `claude_optim.md` sec 4.1 | the Evie-KF operator and its flags | Section 3; Algorithm 1 |
| `claude_optim.md` sec 4.3.1 | the split-batch noise estimate | Proposition 3; Appendix C |
| `claude_optim.md` sec 4.3.3 | the dense-vs-factored self-check | Appendix E; `tests/test_derivation.py` |
| `claude_optim.md` sec 4.3.4–4.3.5 | the descriptive telemetry | Appendix G.4, Table 12 |
| `DERIVATION.md` sec 5, 5.1 | trace normalisation, and why it confounds the diagonal contrast | Appendix G.3; Section 6 |
| `DERIVATION.md` sec 6 | norm preservation | Proposition 4(iii); Appendix I.2 |
| `evie-kf-full-specification.md` | the operator's full specification | Section 3 and Appendix C |
| `kaggle_evie_v3.py` | the validated flat-vector reference implementation the numerics were ground-truthed against | `tests/test_eviekf.py::test_parity_vs_kaggle_reference` |
| `FUTURE_WORK.md` FW-*n* | open questions logged during development | Appendix I.2, *Limitations* |
| `RUNBOOK.md` | how to run a task end to end | shipped, in this directory |

The two `FW-` items that the paper reports measurements for are FW-9, the
mean-eigenvalue normalisation of `gamma` (Section 6 and Appendix I.2), and
FW-10, the norm-preservation ablation (Appendix I.2).
