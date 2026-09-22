# ATOMs in CARLA: attention-based OOD detection for a driving agent

Code for **Chapter 6 (the CARLA study)** of the Master's thesis

> **Monitoring the Attention of Deep Reinforcement Learning Agents for Generalized Out-Of-Distribution Detection**
> Paul Kull, M.Sc. Informatik, Universität Leipzig, 2026

The thesis itself and the Atari study (Chapter 5) are in
[`pk274/ATOMsAD`](https://github.com/pk274/ATOMsAD).

This repository is a fork of **PCLA** (Pretrained CARLA Leaderboard Agents, Tehrani, Kim
and Tonella, FSE 2025), which provides the pretrained driving agents and runs them in
CARLA. PCLA's own README is kept as [`PCLA_README.md`](PCLA_README.md), and its license
(Apache 2.0) applies to the framework. The thesis code is `ATOMs_Analysis/` and the
scripts in the repository root.

**Contents**

1. [The experiment in brief](#1-the-experiment-in-brief)
2. [Reproducing the thesis figures and numbers](#2-reproducing-the-thesis-figures-and-numbers)
3. [Repository layout](#3-repository-layout)
4. [Re-running the pipeline](#4-re-running-the-pipeline)
5. [Archive and removed material](#5-archive-and-removed-material)

---

## 1. The experiment in brief

| | |
|---|---|
| Agent | TransFuser v6 (TFV6) from the LEAD project, pretrained by imitation learning, six cameras. It is the only agent the thesis uses. |
| Relevance | AttnLRP (`ATOMs_Analysis/saliency/lrp_transfuser.py`): one pass from the planning decoder's speed query back to the camera pixels, seeded with its positive activations (profile mode 2, `documentation/04_atoms.md`) |
| Attention profile | the share of relevance on each of ten grouped semantic classes (`ATOMs_Analysis/saliency/atoms_carla.py`) |
| Data | driving frames from the LEAD dataset (`ln2697/lead360` on Hugging Face), split at route level: 5000 reference frames from 186 routes, and a validation (38 routes) and a test split (36 routes) of 1000 frames each |
| Reference | a GMM on the reference profiles. `K = 10` components, chosen on the validation split from `K = 1` to `20`. |
| Scores | Mahalanobis (MD), Euclidean (ED), `k`-NN and Jensen-Shannon (JSD) distance to the reference, and the comparison scores MDX and PEOC |
| Perturbations | brightness increase (`brightness_scale`), camera loss, Gaussian noise, and a PGD attack towards braking. Each evaluation split holds 200 clean frames and 200 of each perturbation. |
| Evaluation | test AUROC per perturbation, route-level bootstrap intervals, and three live runs in CARLA in which a perturbation is switched on mid-drive |

This is the "alternative" split (`EXPERIMENT_VARIANT = "alternative"` in
`ATOMs_Analysis/atoms_config.py`) with profile mode 2 (`MODE_ANALYSIS = 2`), the
configuration every thesis number comes from.

---

## 2. Reproducing the thesis figures and numbers

The figures and numbers are computed from derived data (profiles, labels, fitted
references, per-`K` results) together with the camera frames a few figures show. These
files are too large for git, so they are handed in as a **separate data folder**,
`Atoms_Carla_thesis_data/` (7.4 GB, 256 files). Its `MANIFEST.tsv` lists every file with
its size and SHA-256.

1. **Data.** Copy the `data/` folder from `Atoms_Carla_thesis_data/` into the root of this
   repository. It merges into `data/`, which git ignores.
2. **Environment.** Any Python 3.10+ with numpy, scipy, scikit-learn and matplotlib, for
   example the `atoms3` environment of the Atari repository (`environment.yml` there).
   Torch, CARLA, a GPU and the agent weights are **not** needed. The `environment.yml`
   of this repository is PCLA's environment for running the agents.
3. **Run** [`notebooks/reproduce_thesis_carla.ipynb`](notebooks/reproduce_thesis_carla.ipynb),
   which takes about 5 minutes. Alternatively, run:

   ```bash
   python make_thesis_figures.py --out-dir thesis_figures_reproduced   # 13 figures + .txt sidecars
   python bootstrap_auroc.py     --out-dir thesis_figures_reproduced   # route-level intervals
   ```

   Without `--out-dir` both write to `thesis_figures/`, the committed copies the thesis
   was built from.

The notebook regenerates everything into the git-ignored `thesis_figures_reproduced/`. It
then compares each figure's `.txt` sidecar (the exact numbers behind the figure) with
the committed one, checks the thesis's CARLA images against `thesis_figures/` when the
Atari repository is checked out next to this one, and recomputes the headline numbers of
Section 6.2.

Every sidecar reproduces byte for byte. The PNGs can differ in the font, because
matplotlib takes the first available serif font from the stack in `thesis_style.py`.

### Figure map

| Thesis label | File in `thesis_figures/` |
|---|---|
| `@carlaPca` | `pca_baseline_run_vs_gmm` |
| `@attentions` | `attention_by_cluster` |
| `@valSetSelection` | `auroc_val_test_vs_K` |
| `@aurocs` | `auroc_per_perturbation_gmm` |
| `@distributionShift` | `score_dist_per_perturbation` |
| `@liveBright` | `live_scores_brightness_scale` |
| `@attentionPerCluster` (appendix) | `attention_per_cluster_frames` |
| `@perturbationVsK` (appendix) | `auroc_per_perturbation_vs_K` |
| `@knnSelection` (appendix) | `knn_k_selection` |
| `@clusterAdvantage` (appendix) | `auroc_gmm_vs_single` |
| `@liveNoise`, `@livePgd` (appendix) | `live_scores_gaussian_noise`, `live_scores_pgd` |
| intervals and margins in Section 6.2.1 | `auroc_bootstrap.txt` (from `bootstrap_auroc.py`) |
| `@pgdAttack` (Chapter 4) | from `helpful scripts/visualize_pgd_effect.py`, which runs the attack and therefore needs torch, timm and the TFV6 weights |

`gmm_auc_vs_K` and `attention_per_cluster` are generated as well but not used in the
thesis.

---

## 3. Repository layout

```
├── README.md, PCLA_README.md     this file / the upstream framework's README
├── CLAUDE.md                     detailed working notes on the pipeline (architecture, flags, data layout)
├── notebooks/
│   └── reproduce_thesis_carla.ipynb   reproduces Chapter 6 from the data folder
│
├── make_thesis_figures.py        all CARLA thesis figures + sidecars
├── bootstrap_auroc.py            route-level bootstrap intervals
├── thesis_style.py, figure_notes.py   figure style and sidecars (shared with the Atari repository)
├── thesis_figures/               the committed figures the thesis was built from
│
├── ATOMs_Analysis/               the thesis method
│   ├── atoms_config.py           every path and hyperparameter
│   ├── saliency/                 AttnLRP for TFV6 (lrp_transfuser.py), LRP for WoR, ATOMs profiles
│   ├── detection/                data loading, perturbation application, detectors, GMM clustering
│   ├── perturbation_manager.py   the perturbations
│   └── utils/                    distances, plotting, LRP test suites
├── run_analysis.py               offline analysis: reference fit, scoring, AUROCs for one K
├── sweep_clusters.py             run_analysis.py for every K
├── summarize_results.py          cross-K summary report
├── run_online_analysis.py        scores the live CARLA runs
├── migrate_lead_to_baseline.py   LEAD footage -> reference/test/validation frames
├── apply_val_perturbations.py    perturbs the local validation set
├── cache_live_mdx_scores.py      caches MDX scores of the live runs (needs torch and timm)
├── fix_baseline_leak.py          record of the one-off 2026-09-20 data repair
├── hpc/                          SLURM jobs for the profile computation on the MPCDF Viper cluster
├── helpful scripts/              PGD epsilon sweep, the PGD example figure, saliency plots, GIFs
├── docs/                         design decisions, LRP decision log, cluster how-to
├── documentation/                per-topic code documentation (snapshot from June 2026)
├── documents/                    thesis figure style specification and agent notes
│
├── PCLA.py, pcla_agents/, pcla_functions/, leaderboard_codes/, map_manupulation/,
│   sample.py, route_*.xml, agents.json          the PCLA framework (agents, CARLA interface)
└── archive/                      superseded material (Section 5)
```

---

## 4. Re-running the pipeline

The full pipeline needs the LEAD footage (about 350 GB for all towns), CARLA 0.9.16 on
Linux for the live runs, the TFV6 weights (`python pcla_functions/download_weights.py`)
and a SLURM cluster for the relevance computation (the jobs in `hpc/` were written for
MPCDF Viper). The steps, each documented in more
detail in `CLAUDE.md` and `docs/cluster_explanations.md`, are:

1. **Frames.** Extract the chosen LEAD routes, then write the three splits with
   `python migrate_lead_to_baseline.py --lead_dir <extracted routes> --mode alt_split
   --baseline_n 5000 --test_n 1000 --val_n 1000`. The migration does not clear `frames/` first, and a stale route left
   there caused the leak that `fix_baseline_leak.py` repaired.
2. **Profiles on the cluster.** `hpc/submit_baseline.sh`, `hpc/submit_test.sh` (applies the
   perturbation mix including PGD) and `hpc/submit_val.sh`, each a prep -> array ->
   gather chain. `hpc/compute_mdx_features.py` extracts the MDX features.
   `hpc/collect_results.sh` brings the results back.
3. **Offline analysis.** `python sweep_clusters.py` runs `run_analysis.py` for every `K`
   and writes `data/TFV6/results_alt/<K> clusters/`. `python summarize_results.py`
   writes the cross-`K` report.
4. **Live runs.** Record in CARLA with `LIVE_PERTURBATION_RECORDING_MODE = True`
   (`sample.py` with the TFV6 live-perturbation agent), compute profiles with
   `hpc/submit_live_pert.sh`, then `python cache_live_mdx_scores.py`.
5. **Figures.** Section 2.

---

## 5. Archive and removed material

The repository was cleaned up for the hand-in on 2026-09-22. The state before, which
holds the leak-fixed `K = 10` results the thesis reports, is tagged **`pre-handin-cleanup`**.

* `archive/` holds the World-on-Rails cluster jobs (WoR was the agent of an earlier stage
  and is not used in the thesis) and three stale results summaries. See
  `archive/README.md`.
* The tracked WoR data (`data/WOR/`, 1.1 GB), the older live-run arrays of the original
  Town05 split (`data/TFV6/test_data/`) and an unused LEAD example route
  (`dataset_example_folder/`) are no longer tracked. The WoR agent code
  (`pcla_agents/wor/`) and its LRP support in `ATOMs_Analysis` remain, because they are
  part of the framework. Restore anything with
  `git checkout pre-handin-cleanup -- <path>`.
