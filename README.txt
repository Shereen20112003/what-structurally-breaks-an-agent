WHAT STRUCTURALLY BREAKS AN AGENT?
Controlled Diagnostics for Stateful Workflows

paper.pdf is the authored full manuscript with appendix.
review.pdf is the anonymous short version.
workshop.pdf is the anonymous workshop-length version.
supplement.pdf is the authored appendix/supplement.
Editable sources: paper.tex, workshop.tex, review.tex, supplement.tex.

RESULTS
49,152 seeded tasks, 393,216 reference-controller episodes, 192 factorial
cells, eight local reference policies, 256 paired seeds. All tasks have
eight required operations. Exact solves every task. No language-model
model runs are included in these two reference-controller suites.

The separate follow-on design has 1,156 configurations and 295,936
additional reference-policy episodes, using disjoint seeds 10,000-10,255.
Seven study groups test work/operation count, retention strategies,
snapshot cadence and hidden resampling, recovery/retries, query locality,
selection cost, and label reversal. Total local reference-policy episodes:
689,152. These are programmed controllers, not model experiments.

The main finding is methodological: precedence depth changes scheduling
freedom even at fixed total work. Reference controllers distinguish
scheduling, candidate-checking, and fixed-cache retention limitations.
The snapshot/history intervention removes state refreshes, not information
recoverable from history. Do not describe it as irrecoverable uncertainty.

MODEL PILOT
An additional exploratory, model-in-the-loop pilot contains 24 unique
episodes: one Copilot Auto-routed GPT-6 Luna model, four paired low/high
interventions, and three task seeds per cell. Sixteen episodes use the two
initial seeds; eight episodes use a third seed added after inspecting those
results. The raw log contains 29 records; five repeated/recovery records
are excluded from summaries. Results are descriptive only. No Gemini or
Qwen hosted route, selectable reasoning-budget condition, realistic transfer
task, or confirmatory multi-model comparison was available. Provider usage
logs record 168 premium-request units including the smoke test and 16.180
billion nano-AIU; no dollar value is reported by the CLI record.

COPILOT FRACTIONAL-FACTORIAL FOLLOW-UP
A separately frozen 2^(4-1) resolution-IV half fraction evaluated the same
Copilot-routed GPT-6 Luna interface on eight task configurations and two
paired seeds (16 episodes, eight required operations each). It yielded
8/16 successes and 88/128 completed operations; seven episodes failed on
candidate selection and one on dependency order. Descriptive high-minus-low
success contrasts were depth -25 pp, branching +50 pp, width +25 pp, and
snapshot feedback +50 pp. There are only two seeds; main effects are
potentially aliased with three-factor interactions, and pairwise
interactions are mutually aliased. The final planned episode was completed
after inspecting the preceding 15 outcomes following a pause; the design
was unchanged. These results are not confirmatory and do not establish a
general capability profile. See model_factorial_protocol.json.
The run used 96 logged model calls; across recorded Copilot activity, the
usage files contain 264 premium-request units and 25.653 billion nano-AIU.
Copilot did not report a USD equivalent.

LOCAL MODEL STUDY
A separate, no-API CPU study uses Qwen/Qwen2.5-0.5B-Instruct-GGUF, Q4_K_M,
with a complete 2^5 factorial (256 planned episodes, eight paired seeds,
eight required operations). It varies chains/barriers, depth 2/8, branching
2/8, register width 1/8, and history/snapshot feedback. The model file is
identified by SHA-256 in local_model_protocol.json. The runtime is
llama.cpp b12030 CPU x64; its official release asset SHA-256 is recorded in
the protocol. The smoke episode is stored separately and excluded. No
hosted model calls or reasoning-budget comparison are involved. Only a
complete, unique 256-episode grid is accepted by the analysis script.
Observed outcome: 0/256 workflows completed; the model completed 21/2,048
required operations, and 255/277 decisions were strictly schema-valid JSON.
All success contrasts are zero because of this floor effect; they do not
show that the task factors have no effect. This single-model run does not
yield a structural capability profile.

For a fresh reproduction, obtain the exact model and runtime assets noted in
local_model_protocol.json, verify their hashes, then use:

.\.venv\Scripts\python.exe run_local_study.py --smoke
.\.venv\Scripts\python.exe run_local_study.py
.\.venv\Scripts\python.exe analyze_local_study.py

REPRODUCE ON WINDOWS
Open this folder in VS Code; use a PowerShell terminal:

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
.\.venv\Scripts\python.exe run_experiments.py --seeds 256
.\.venv\Scripts\python.exe analyze.py
.\.venv\Scripts\python.exe run_extended.py
.\.venv\Scripts\python.exe analyze_extended.py
.\.venv\Scripts\python.exe analyze_model_pilot.py
.\.venv\Scripts\python.exe analyze_model_factorial.py
.\.venv\Scripts\python.exe analyze_local_study.py
.\build.ps1
.\.venv\Scripts\python.exe verify_artifact.py
.\.venv\Scripts\python.exe package_artifact.py

The reference-controller benchmark and analysis need only the Python standard library.
pypdf and PyMuPDF support document extraction/layout validation.
The installed local tools\tectonic.exe builds the manuscript; the first
build may download public TeX packages. A standard LaTeX installation with
acmart and pgfplots can also compile these sources.

Changing the primary seed count requires an explicit protocol revision.
The experiment runner rejects a mismatched primary seed count.
Protocol status is a locally frozen design, not external preregistration.
Wilson scheduling intervals were added after the primary result inspection.

ARTIFACT
results\episodes.csv: every task/controller outcome and exact conditional
prediction, plus failure category, progress, calls, and candidate checks.
results\structural_audit.csv: graph metrics for every factorial cell.
results\metadata.json: execution metadata and experiment source hashes.
results\analysis.json: summaries, contrasts, interactions, analysis hash.
results\curves.csv: all marginal profiles for both graph families.
results\scheduler_validation.csv: 256 unique-seed scheduling trials per
depth/family and Wilson intervals; repeated structural conditions are not
independent extra trials.
generated\: data-derived LaTeX tables and plot coordinates.
results_model_pilot\: paired action traces, per-response Copilot usage
records, exploratory cell summaries and the generated manuscript table.
results_model_factorial\: frozen fractional-factorial protocol, 16 episode
traces, interrupted-attempt checkpoint, 96 per-call usage records, analysis,
and generated manuscript table.

PACKAGE
paper-source.zip: manuscript, supplement and all LaTeX inputs.
reproducibility.zip: manuscript sources, experiment code, protocol, results,
instructions and verification records. Excludes .venv, compiler binaries,
the supplied proposal PDF, extracted text and local preview images.
No public repository or archive has been created.

SUBMISSION STATUS
The user requested anonymous authorship for now. This is not a camera-ready
paper and should not be uploaded to arXiv anonymously. See SUBMISSION.txt.
Provisional target: Adaptive and Learning Agents (ALA), methods/work-in-
progress track, if offered by its upcoming AAMAS edition. Current-edition
workshop approval, CFP, dates, and requirements remain unverified.
No acceptance, endorsement, external validity, cross-provider result, or
reasoning-budget comparison is claimed. Related work demonstrates that
controllable complexity, memory-demand design, tool-trajectory diagnosis,
multi-step tool benchmarks, and budget evaluation are established areas.
TRAJECT-Bench, Reward Hacking Benchmark, SynthAPI, and tunable-memory POMDPs
are close precedents. The only candidate distinction now identified is a
fixed-work, state-mutating workflow with audits of recoverable state and
controller representation; this is not yet a demonstrated novel contribution.
Model evidence consists of the post-inspection 3-seed Copilot pilot, the
two-seed Copilot half-fraction, and one small quantized local model at the
success floor. This is not cross-model generalization.

CREDITS AND MODEL ACCESS
Recorded Copilot usage is 25.653 billion nano-AIU and 264 premium-request
units across the smoke and model-pilot studies, including the fractional
factorial. These are Copilot usage meters, not USD; the monthly quota and
dollar equivalent cannot be read from this environment.
Copilot CLI model choice was restricted to Auto-routed GPT-6 Luna in this account;
explicit Gemini and GPT model identifiers were rejected, and no Qwen
endpoint or provider credentials were configured. The Auto route did not accept a reasoning-effort setting. The completed
fractional factorial remains exploratory and incomplete relative to a full
factorial, by design; no further hosted inference is included.
