# What Structurally Breaks an Agent? Auditing Difficulty in Stateful Tool-Use Workflows

Executable diagnostic instrument and manuscript for studying scheduling, candidate selection, and state retention in fixed-work tool-use workflows. This repository contains the current paper source, reference-controller experiments, and exploratory model checks. It is an unpublished draft, not a conference acceptance or a general claim about deployed agents.

## Authors

The work was done while Shereen Anand was at Amazon. She is currently at Microsoft Research; this idea is independent of that affiliation.

- Shereen Anand, Amazon — shereenv@amazon.com
- Ju Wang, University College London
- Deepanway Ghosal, Google DeepMind, India
- Yali Du, King's College London — yali.du@kcl.ac.uk
- R. Venkatesh Babu, Indian Institute of Science — venky@iisc.ac.in

Public addresses are included only where they were verified. Ju Wang and Deepanway Ghosal are listed without an email because no current public address was confirmed. Author order should be confirmed by the authors before circulation.

## Manuscript

- `paper.tex` is the editable full manuscript, including the author block above.
- `paper.pdf` is the compiled manuscript. Regenerate it after editing `paper.tex`; an older PDF will not show later author changes.
- `workshop.tex` and `review.tex` are anonymous short versions.
- `supplement.tex` and `appendix_body.tex` contain the appendix material.
- The ACM class is not bundled. Compile `paper.tex` with the ACM `acmart` template.

## What the experiments establish

The primary reference-controller study contains 49,152 seeded tasks and 393,216 episodes. A disjoint follow-on design contains 1,156 configurations and 295,936 episodes. These are programmed controllers, not language-model runs. Exact reference controllers solve the generated tasks. The central methodological result is that increasing precedence depth at a fixed operation count also changes scheduling freedom, so the controls are not universally independent.

The model checks are exploratory and are not confirmatory capability results:

- A 24-episode Copilot-routed pilot and a 16-episode, two-seed fractional factorial use one routed model. Their contrasts are descriptive.
- A separate local CPU study of one quantized 0.5B model completed none of 256 workflows. That floor result does not show that the task factors have no effect.

No cross-provider comparison, reasoning-budget comparison, or realistic transfer study has been completed.

## Reproducing the local study

The model weights and llama.cpp binaries are intentionally excluded. Obtain the exact assets named in `local_model_protocol.json`, verify their recorded hashes, and then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s . -p "test_*.py"
.\.venv\Scripts\python.exe run_local_study.py --smoke
```

The full local study and analysis are `run_local_study.py` and `analyze_local_study.py`. Do not treat the smoke run as a scientific result.
