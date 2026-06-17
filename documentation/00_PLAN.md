# Documentation Sweep — Plan

Goal: systematically review the repo and produce one technical-reference `.md` per subtopic in `documentation/`, as source material for the thesis implementation chapter. Style: dense factual notes — every design decision with rationale and file/line references. Scope: TFV6 primary, WoR documented where it differs. Bugs are flagged (not fixed) in a shared bug log. Language: English.

## Topic files (in review order)

The order follows the data-flow of the pipeline, so each topic can reference concepts already documented.

| # | File | Content | Primary sources |
|---|------|---------|-----------------|
| 1 | `01_architecture_overview.md` | Repo layers (PCLA vs ATOMs_Analysis), end-to-end data flow (raw footage → npz → profiles → detectors → results), config system: `atoms_config.py` as single source of truth, `AGENT`, `MODE_ANALYSIS`, `EXPERIMENT_VARIANT`, recompute flags, path resolution | `atoms_config.py`, `CLAUDE.md`, repo layout |
| 2 | `02_agents.md` | TFV6 architecture (encoders, GPT fusion, PlanningDecoder, speed bins/two-hot), choice of F_c = `speed_query`; WoR `CameraModel` and its F_c; sensor/input formats | `pcla_agents/transfuserv6/`, `pcla_agents/wor/`, `docs/lrp_todo.md` |
| 3 | `03_lrp.md` | AttnLRP for TFV6 (`lrp_transfuser.py`): custom autograd for softmax/matmul, ε/AlphaBeta rules, `TFv6FullModelForLRP`, two-pass scheme (LRP1/LRP2/output→input), softmax-distribution seed rationale; WoR z⁺-rule (`lrp_analysis.py`); status of `lrp_lbc.py`; open issues (BatchNorm canonization, contrastive seeding) | `saliency/lrp_transfuser.py`, `saliency/lrp_analysis.py`, `docs/lrp_todo.md`, `papers/attention lrp.pdf` |
| 4 | `04_atoms.md` | `ATOMsCarla`: hierarchical h(o) (R̄ + renormalization) and combinatorial c(T) (α=0.25), class sets (29-dim WoR vs 10-dim grouped TFV6), MODE_ANALYSIS 1 vs 2, `_make_minimal_data` reconstruction and its limits (zero target_point/acceleration), comparative-relevance slots | `saliency/atoms_carla.py`, `papers/ATOM-paper.pdf`, `docs/design_decisions.md` |
| 5 | `05_dataset_creation.md` | Baseline / test / val set creation: route selection (`unzip_routes.ps1` logic), Town05 holdout rationale, `migrate_lead_to_baseline.py` (sampling, command mapping, npz schema), `alt_split` same-distribution variant, val-set auto-exclusion; `baseline_dataset.py` and `dataset.py` loaders | `migrate_lead_to_baseline.py`, `detection/baseline_dataset.py`, `detection/dataset.py` |
| 6 | `06_perturbations.md` | `PerturbationManager` registry, each perturbation (gaussian_noise, brightness_scale, camera_loss, pgd) with parameters and design rationale, `PerturbationApplier` / `PerturbationSpec`, labeled test/val set construction (5-way 20% mix), live-perturbation variants | `perturbation_manager.py`, `detection/dataset.py` |
| 7 | `07_distances_and_detectors.md` | `DistanceComputer` (stateless metrics); each detector: Mahalanobis (single + GMM), Euclidean, k-NN (k selection via val AUC), JSD, Wasserstein, MDX-v1 vs MDX-v2 (feature spaces, binning strategies), Action Entropy, PEOC; `GMMClustering` BIC/AIC sweep and K-resolution priority; threshold policy (99th percentile, Youden-J) | `detection/detectors.py`, `detection/clustering.py`, `utils/distance_computer.py` |
| 8 | `08_offline_analysis.md` | `run_analysis.py` step-by-step (steps 1–12 incl. 2.5/2.5-v2/9.5), val-set role, per-perturbation breakdown, disabled trajectory analysis (step 8.5), `sweep_clusters.py`, `summarize_results.py` and SUMMARY.md generation, results layout | `run_analysis.py`, `sweep_clusters.py`, `summarize_results.py` |
| 9 | `09_online_analysis.md` | `run_online_analysis.py`: live perturbation injection mid-drive, what is measured, windowing/score smoothing if any, relation to offline detectors, live_pert data layout | `run_online_analysis.py`, `hpc/prep_live_pert*.py` |
| 10 | `10_hpc_pipeline.md` | Viper workflow: chunked profile computation, array jobs, gather scripts, MDX feature precomputation, sync; what runs on HPC vs locally and why | `hpc/`, `docs/cluster_explanations.md` |
| 11 | `11_validation_and_testing.md` | LRP/ATOMs correctness validation: `lrp_test_suite.py`, `tfv6_test_suite.py`, `atoms_test_suite.py`, diagnostics scripts; which properties are checked (conservation, sanity checks) — valuable for the thesis "validation of the implementation" section | `utils/*test_suite*.py`, `utils/*diagnostics*.py` |
| 12 | `12_visualization.md` | Figure inventory: what each plot type shows, `viz_config.py` thesis-wide styling, which figures map to which thesis sections | `utils/visualization_carla.py`, `utils/viz_config.py` |
| — | `99_bugs_and_findings.md` | Running log: every bug/inconsistency/stale-doc found during the sweep, with file/line, severity, suggested fix. No code changes. | filled by all topic reviews |

## Method

- One subagent per topic, run sequentially (one topic per step, you confirm before the next). Each subagent reads the listed sources end-to-end, cross-checks against `docs/design_decisions.md` / `docs/lrp_todo.md` / `CLAUDE.md` for staleness, and writes its topic file plus bug-log entries.
- Each topic file gets a fixed structure: Purpose → Key design decisions (with rationale + alternatives considered where discoverable) → Implementation details (file/line refs) → Parameters/configuration → Known limitations & open issues → Cross-references.
- After all topics: a consistency pass over the whole `documentation/` folder (cross-references, contradictions, terminology).

## Resolved questions

1. MDX is based on Zhang ("A Simple Unified Framework for AD in DRL"); PEOC on Müller/Sedlmeier (uncertainty paper). Detectors are documented against these originals.
2. `lrp_lbc.py` is out of scope (dead end).
3. Atari experiment: out of scope (separate repo/sweep).
4. Existing `docs/*.md` are cross-checked but not trusted — everything verified against code.
5. Alternative split is documented as a first-class variant, with pros/cons of both splits.
6. **All magic constants** (hardcoded thresholds, percentiles, sample counts, α values, ε values, bin edges, seeds, etc.) must be captured in each topic's Parameters section, marked as hardcoded vs configurable.

## Order of execution

Steps 1–12 as numbered above. Rationale: config/architecture first (everything references it), then the saliency core (LRP → ATOMs), then data (sets → perturbations), then detection (distances → offline → online), then infrastructure (HPC, testing, visualization). The bug log accumulates throughout.
