# Bounded ATMS, Baselines, OpenRouter Pricing, and CLUTRR Splits Survey

## Summary

Phase 1 — ATMS: The best Python ATMS is Ruggiero-Santo/NATMS (GitHub), implementing de Kleer's labeling algorithm with negation. Labels are set[frozenset]; environments are frozensets of assumption names. Missing features that must be added: (a) depth cap — filter environments where len(E) > d; (b) size cap on label cardinality; (c) indexed no-good lookup. Pseudocode for bounded label propagation is provided in research_out.json. De Kleer 1992 (AAAI) describes incremental prime-implicate generation but no explicit bounded variant; depth=len(E) is the natural approximation.

Phase 2 — Baselines: LINC (benlipkin/linc, EMNLP 2023) uses Prover9 FOL prover; entry point is `accelerate launch runner.py`; LLM API redirectable to OpenRouter by patching OPENAI_BASE_URL. Logic-LM (teacherpeterpan/Logic-LLM, EMNLP 2023) uses SMT solver (Z3 via SatLM); two-step pipeline (logic_program.py + logic_inference.py); LLM swappable via OpenAIModel in utils.py. Path-of-Thoughts (arXiv:2412.17963) has NO public code; three-stage reimplementation spec provided (graph extraction via LLM, DFS path finding, per-path reasoning via LLM or ASP/CLINGO). ProbLog (ML-KULeuven/problog): 5-line Python API confirmed — PrologString + get_evaluatable().evaluate() returns query probabilities.

Phase 3 — OpenRouter: deepseek/deepseek-v4-flash at $0.0983/$0.1966 per M tokens (1M ctx) is recommended for L1 JSON extraction. deepseek/deepseek-v3.2 at $0.2288/$0.3432 per M is recommended for L2 assumption proposal. Total budget estimate ~$0.86 for 400 documents × all baseline runs, leaving 11.6× headroom under $10.

Phase 4 — CLUTRR: Test set has 1,857 instances across k=2-10: k=2(114), k=3(229), k=4(219), k=5(308), k=6(178), k=7(246), k=8(228), k=9(172), k=10(163). HuggingFace load: load_dataset('CLUTRR/v1', 'gen_train23_test2to10', trust_remote_code=True). Stratified 200-instance subsample: 9 strata × 22 = 198; all strata feasible (minimum available 114).

## Research Findings

## Phase 1: Python ATMS Implementations

Two Python ATMS implementations exist on GitHub. The best is **Ruggiero-Santo/NATMS** [2] — a negation-extended ATMS based on de Kleer's general labeling algorithm. It implements `UPDATE`, `WEAVE`, `PROPAGATE`, and `NOGOOD` functions. Labels are `set[frozenset]`; environments are `frozenset` of assumption names [2]. The secondary option is **FellnerDotDev/ATMS-in-Python** [1], but its README explicitly states the algorithm is incomplete.

Neither repo implements a **depth cap** (filter environments where `len(E) > d`) or a **size cap** (bound label cardinality). Both must be added. De Kleer (1992) [17] describes incremental prime-implicate generation for ATMS but does not specify a simple depth bound — the natural approximation is `depth(E) = len(E)`, the assumption-set size. Concrete pseudocode for the bounded label-propagation loop is provided in `research_out.json`.

## Phase 2: Baseline Interfaces

**LINC** [3] (EMNLP 2023): Entry point `accelerate launch runner.py`; FOL prover is **Prover9** (must install separately). Config in `linc/eval/args.py`. LLM API redirectable to OpenRouter by setting `OPENAI_BASE_URL=https://openrouter.ai/api/v1`. Custom datasets require format matching FOLIO/ProofWriter JSON schema.

**Logic-LM** [4] (EMNLP 2023): Two-step pipeline — `models/logic_program.py` (generation) then `models/logic_inference.py` (inference). Symbolic solver is **SMT/Z3-based** (code from SatLM). LLM swappable by modifying `OpenAIModel` class in `utils.py`. Custom datasets: add `./models/prompts/{dataset_name}.txt` with `[[PROBLEM]]`/`[[QUESTION]]` placeholders.

**Path-of-Thoughts** [5, 6]: **No public code released** as of June 2026. Must reimplement from paper. Three stages: (1) single LLM call to extract entity-relation-entity triples and build graph; (2) deterministic DFS to find all source→target paths; (3) per-path LLM inference (PoT-LLM) or ASP/CLINGO symbolic reasoning (PoT-Symbolic). Paper uses CLUTRR test set with 1,049 samples (k=2–10), stories modified to remove name tags. Achieves up to 21.3% improvement over SOTA [6].

**ProbLog** [7, 8]: `pip install problog`. Minimal Python API — 5 lines:
```python
from problog.program import PrologString
from problog import get_evaluatable
model = "0.8::parent(tom,bob). 0.9::parent(bob,ann). ancestor(X,Y):-parent(X,Y). query(ancestor(tom,ann))."
result = get_evaluatable().create_from(PrologString(model)).evaluate()
# => {ancestor(tom,ann): 0.72}
```
Probabilistic facts use `p::fact.` syntax; `query(term).` marks evaluation targets [8].

## Phase 3: OpenRouter Models

Confirmed pricing (June 2026): **deepseek/deepseek-v4-flash** [9]: $0.0983 input / $0.1966 output per M tokens; 1M context window; MoE 284B/13B parameters. **deepseek/deepseek-v3.2** [10]: $0.2288/$0.3432 per M tokens; 131K context. **google/gemini-2.5-flash** [11]: $0.30/$2.50 per M (expensive output). **meta-llama/llama-3.3-70b-instruct** [12]: $0.10/$0.32 per M; free tier available.

Recommended: L1 grounding → `deepseek/deepseek-v4-flash`; L2 assumption proposal → `deepseek/deepseek-v3.2`.

Budget estimate [9, 10, 12]: 400 docs × (2 L1 calls at $0.087 + 1 L2 call at $0.128) + 3× baseline overhead ≈ **$0.86 total** against a $10 budget → 11.6× headroom [9, 10].

## Phase 4: CLUTRR Dataset

Per-hop test instance counts [14, 16]: k=2(114), k=3(229), k=4(219), k=5(308), k=6(178), k=7(246), k=8(228), k=9(172), k=10(163) — total 1,857 test instances. Training covers k=2,3 (or k=2,3,4); testing always spans k=2–10 to evaluate generalization [14].

HuggingFace load [15]: `load_dataset('CLUTRR/v1', 'gen_train23_test2to10', trust_remote_code=True)['test']`. Six total configurations including robustness variants.

Stratified 200-instance subsample [16]: 9 strata (k=2–10) × 22 per stratum = 198 instances. All strata feasible — minimum available is 114 (k=2). Stratified sampling code provided in `research_out.json`.

## Sources

[1] [FellnerDotDev/ATMS-in-Python](https://github.com/FellnerDotDev/ATMS-in-Python) — Python 3 ATMS based on de Kleer; described as incomplete; file atms_algo.py

[2] [Ruggiero-Santo/NATMS — Negation ATMS](https://github.com/Ruggiero-Santo/NATMS) — Best Python ATMS; implements UPDATE/WEAVE/PROPAGATE/NOGOOD; labels as set[frozenset]; no depth cap

[3] [benlipkin/linc — LINC EMNLP 2023](https://github.com/benlipkin/linc) — LINC repo; entry point runner.py; Prover9 FOL prover; eval/args.py for CLI

[4] [teacherpeterpan/Logic-LLM — Logic-LM EMNLP 2023](https://github.com/teacherpeterpan/Logic-LLM) — Logic-LM; logic_program.py + logic_inference.py; SMT/Z3 solver; OpenAIModel in utils.py for LLM

[5] [Path-of-Thoughts (arXiv:2412.17963)](https://arxiv.org/abs/2412.17963) — PoT paper; no public code; three-stage pipeline; CLUTRR 1049 test samples k=2-10

[6] [PoT paper HTML — pipeline detail](https://arxiv.org/html/2412.17963) — Stage specs: LLM graph extraction, DFS path finding, per-path LLM/ASP reasoning; 21.3% gain

[7] [ML-KULeuven/problog](https://github.com/ML-KULeuven/problog) — ProbLog GitHub; pip install problog; Python API confirmed

[8] [ProbLog Python API docs](https://problog.readthedocs.io/en/latest/python.html) — Minimal: get_evaluatable().create_from(PrologString(model)).evaluate() returns probability dict

[9] [OpenRouter deepseek/deepseek-v4-flash](https://openrouter.ai/deepseek/deepseek-v4-flash) — $0.0983/$0.1966 per M tokens; 1M ctx; MoE 284B/13B; released 2026-04-24

[10] [OpenRouter deepseek/deepseek-v3.2](https://openrouter.ai/deepseek/deepseek-v3.2) — $0.2288/$0.3432 per M tokens; 131K ctx; strong reasoning + tool use

[11] [OpenRouter google/gemini-2.5-flash](https://openrouter.ai/google/gemini-2.5-flash) — $0.30/$2.50 per M tokens; 1M ctx; built-in thinking; expensive output

[12] [CostGoat OpenRouter Pricing Guide (Jun 2026)](https://costgoat.com/pricing/openrouter) — Confirms deepseek-v4-flash ~$0.10/$0.20; llama-3.3-70b $0.10/$0.32; free models available

[13] [facebookresearch/clutrr](https://github.com/facebookresearch/clutrr) — CLUTRR official repo; 22 kinship relations; k-hop difficulty; Google Drive pre-generated data

[14] [CLUTRR: A Diagnostic Benchmark (arXiv:1908.06177)](https://arxiv.org/pdf/1908.06177) — Original paper; train k=2,3 or k=2,3,4; test k=2-10; 6,016 AMT paraphrases

[15] [CLUTRR HuggingFace Dataset](https://github.com/kliang5/CLUTRR_huggingface_dataset) — 14 configurations; gen_train23_test2to10 is standard; 8-12k train, 400-1100 test per config

[16] [Graph-based Synthetic Data Augmentation (arXiv:2409.12437)](https://arxiv.org/pdf/2409.12437) — Table 2: per-hop test counts k=2(114) to k=10(163); total 1,857 test instances

[17] [de Kleer 1992: Improved Incremental Algorithm for Prime Implicates (AAAI)](https://link.springer.com/chapter/10.1007/978-3-642-60211-5_9) — Incremental prime implicate generation for ATMS; NP-complete; exploits prior computation

[18] [OpenRouter Models Catalog](https://openrouter.ai/models) — 315+ models; free tier (rate limited); structured output support queryable via API

## Follow-up Questions

- PoT reimplementation needs the exact LLM prompts from paper Appendix A.4-A.6. Should the executor use a generic extraction prompt or attempt to retrieve the full appendix from ar5iv.labs.arxiv.org/html/2412.17963v1?
- LINC custom dataset format: if LINC's eval/args.py does not support kinship story format without significant modification, should the executor fall back to a raw Prover9 baseline or skip LINC in favor of Logic-LM alone?
- Is google/gemini-2.5-flash-lite available on OpenRouter (it exists at ~$0.10/M input on Google AI Studio), and would it be a cheaper L2 alternative to deepseek/deepseek-v3.2?

---
*Generated by AI Inventor Pipeline*
