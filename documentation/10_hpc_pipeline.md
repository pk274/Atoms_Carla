# Topic 10 — HPC Pipeline: Viper Chunked Profile Computation, Array Jobs, and the prep→array→gather Decomposition

All claims verified against code on 2026-06-14. Line numbers refer to the current working tree.
Primary sources read in full (under `hpc/`): the four compute workers `compute_baseline_chunk.py` (221), `compute_test_chunk.py` (258), `compute_live_pert_chunk.py` (203), `compute_mdx_features.py` (145); the four prep scripts `prep_test.py` (178), `prep_test_wor.py` (167), `prep_live_pert.py` (90), `prep_live_pert_wor.py` (92); the three gather scripts `gather_baseline.py` (120), `gather_test.py` (111), `gather_live_pert.py` (106); the SLURM glue `array_task.sh`, `array_test_task.sh`, `array_live_pert_task.sh`, `array_task_wor.sh`, `gather_task.sh`, `gather_test_task.sh`, `gather_live_pert_task.sh`, `prep_test_task.sh`, `prep_test_task_wor.sh`, `prep_live_pert_task.sh`, `submit_baseline.sh`, `submit_test.sh`, `submit_val.sh`, `submit_live_pert.sh`, `submit_baseline_wor.sh`; the operator helpers `sync_to_hpc.sh`, `collect_results.sh`, `setup_venv.sh`, `requirements_hpc.txt`, `visualize_perturb.py`; the stubs `stubs/carla.py`, `stubs/beartype.py`. Cross-checked against `ATOMs_Analysis/atoms_config.py`, `docs/cluster_explanations.md`, `CLAUDE.md`, and `documentation/03_lrp.md`, `04_atoms.md`, `06_perturbations.md`, `07_distances_and_detectors.md`, `08_offline_analysis.md`, `09_online_analysis.md`.

---

## 1. Purpose & scope

This document covers the **HPC compute layer** — the `hpc/` directory that offloads the single most expensive step of the project (per-frame LRP + ATOMs profile computation) onto the MPCDF **Viper** SLURM cluster, then ships the small numeric outputs back into the git repo for the local analysis in Topics 8–9 to consume. It is the **infrastructure that produces the profiles** the detection pipeline scores; it does not itself run any detector, fit any Gaussian/GMM, or draw any ROC. The division is deliberate and is the central design fact of this topic: **per-frame attribution runs on HPC; orchestration, detector fitting, scoring, evaluation, and figures run locally** (Topic 8 §2.5).

The layer is organised into **three parallel workflows** — baseline, test, live_pert — each available in a **TFV6** and a **WoR** variant, and each decomposed into the same three stages:

1. **prep** (CPU, model-free): build the index-addressable input NPZ — either the labeled test/val set (`prep_test*.py`, Topic 6's spec applied here) or the concatenated live-pert recording (`prep_live_pert*.py`). Writes a `*_meta.txt` frame count used to size the array.
2. **array-compute** (one SLURM array task per chunk): load the agent + LRP wrapper, run LRP + ATOMs over a slice of frames, write a partial `.npz`. This is the only stage that loads the model and is the entire reason the pipeline exists.
3. **gather** (single node, dependency `afterok` on the array): concatenate the partials in frame order into the final `baseline_<mode>.npz` / `test_profiles_<mode>.npy` / `live_pert_profiles_<mode>.npy` (+ PEOC logit arrays + `mdx_features.npz`).

The whole chain is launched by one `submit_*.sh` per workflow, which chains prep→array→gather via SLURM `--dependency=afterok`. Results are returned to the local repo by `collect_results.sh` (`cp` + `git add -f`); frames are pushed to Viper by `sync_to_hpc.sh` (HTTP-over-reverse-tunnel). `docs/cluster_explanations.md` is the human-facing operator guide for the transfer mechanics; this topic documents its workflow and flags where it has drifted from the scripts.

Two load-bearing facts of this review: (a) **the cluster is Viper-CPU, not GPU** — `requirements_hpc.txt:1` says so explicitly, the CPU torch wheels are pinned, and **no SLURM script requests a GPU** (`--ntasks=1 --cpus-per-task=4`, no `--gpus`), so the compute workers' `cuda if available` branch always falls to CPU on Viper (finding 10.1); and (b) **the chunk workers hardcode `p_relevance=0.25`** while the local config default is `0.9` (finding 4.3, confirmed verbatim below), so HPC-computed profiles are not reproducible with the local default — the single most consequential local↔HPC divergence in the project.

This topic does **not** re-derive LRP (Topic 3), the ATOMs profile (Topic 4), the perturbation pixel transforms or the deferred-PGD design (Topic 6), or detector math (Topic 7). It documents the chunking, array sizing, SLURM resources, deferred-PGD crafting *on the chunk worker*, MDX feature precompute, gather ordering, the transfer model, and the magic constants.

---

## 2. Key design decisions

### 2.1 Why HPC at all — the per-frame LRP+ATOMs cost

A single ATOMs profile requires one full LRP backward pass through the agent (AttnLRP for TFV6, z⁺ for WoR; Topic 3) plus the per-class intersection with the segmentation mask (Topic 4). The compute workers self-report ≈ a few frames/second on CPU (their `fr/s` ETA prints, e.g. `compute_baseline_chunk.py:191-195`), so baseline clouds of several thousand frames and labeled test/val sets of ~500–1000 frames are hours of serial work. The HPC layer exists to **parallelise this across SLURM array tasks**: each task is an independent process owning a disjoint slice of frames, so wall-clock time drops to the per-chunk cost. Because the local config gates the expensive stages behind `RECOMPUTE_*=False` (Topic 8 §2.1), the local `run_analysis.py` then *loads* the HPC-produced arrays and never re-runs LRP — the local machine only fits detectors and renders figures, which is seconds-to-minutes.

### 2.2 prep→array→gather, and why prep is a separate model-free CPU job

The pipeline is split into three stages with distinct resource profiles:

- **prep is model-free** so it can run on a cheap CPU node without loading PyTorch weights or CARLA. For the test/val path it applies the image-space perturbations and writes the labeled NPZ (`prep_test.py`); for live_pert it just concatenates recordings (`prep_live_pert.py`). Crucially, **PGD cannot be crafted model-free**, so prep deliberately records `pgd` frames with *clean pixels* + `label=1` + `perturbation="pgd"` and **defers the attack to the array stage** where the model is loaded (`prep_test.py:7-9,131-137`; Topic 6 §2.3). This is the reason prep and array are separate jobs rather than one.
- **array-compute is the only GPU-class stage** (model + LRP) and is the one that fans out across tasks. Splitting it from prep means the array can be re-submitted (e.g. for the other `MODE_ANALYSIS`) **without re-running prep** — `submit_test.sh:78-80` skips prep when `test_labeled.npz` already exists, and the labeled set is mode-independent so both modes share it.
- **gather is a single high-memory node** (`--mem=80000MB`, `gather_task.sh:15`) that loads every partial and concatenates; it needs RAM, not parallelism (`--cpus-per-task=1`).

The dependency chaining (`--dependency=afterok`) guarantees array waits for prep and gather waits for array, so a single `submit_*.sh` call queues the whole chain and the operator only monitors `squeue`.

### 2.3 Chunking for array parallelism — two different chunk granularities

There are **two distinct chunking schemes**:

- **Baseline: one array task per run file** (`submit_baseline.sh:45-68`). The submit script globs `run_*.npz`, writes a `run_file_list.txt`, sets `--array=0-(N_files-1)`, and each task `sed`-indexes its run file (`array_task.sh:33`). The chunk granularity is therefore *one whole route file* (variable frame count), and `compute_baseline_chunk.py` processes all frames in that file (`:142-155`).
- **Test / val / live_pert: one array task per fixed-size frame slice** (`submit_test.sh:59`, `submit_live_pert.sh:44`). The submit script computes `N_TASKS = ceil(N_FRAMES / CHUNK_SIZE)` with `CHUNK_SIZE=20` default, and each task derives its half-open slice `[CHUNK_START, CHUNK_END)` from `SLURM_ARRAY_TASK_ID` (`array_test_task.sh:32-33`). The worker clamps `chunk_end` to `n_total` and writes an **empty partial** for tasks past the end (`compute_test_chunk.py:120,124-134`), which gather later drops (`gather_test.py:73-76`).

The fixed-size scheme is what makes test/live arrays *index-addressable* — the worker slices `data["wide_rgb"][i:i+1]` by absolute frame index, and gather re-orders partials by their stored `chunk_start` so the concatenation matches the original `test_labeled.npz` row order (`gather_test.py:71,78`). Baseline has no `chunk_start`; gather instead orders partials by the numeric array-task-id suffix in the filename (`gather_baseline.py:51-52`), which equals the `run_file_list.txt` order — order is immaterial for the baseline mean/covariance but the `series` rows inherit run-file order.

### 2.4 MDX feature precompute to avoid local GPU

MDX-v1 needs 512-d ResNet34 backbone features per baseline frame (Topic 7 §2.7). Extracting these locally is a per-frame forward pass that `run_analysis.py` warns is "slow" (Topic 8 §3.4). The baseline chunk worker therefore **also extracts the 512-d backbone feature and a speed-derived action proxy alongside the ATOMs profile** (`compute_baseline_chunk.py:184-189` for TFV6, `:166-183` for WoR's 576-d feature + 28-bin action marginals), and `gather_baseline.py` writes them into a separate **`mdx_features.npz`** (`:105-114`). Locally, `run_analysis.py` reads `mdx_features.npz` and fits the MDXDetector in seconds instead of recomputing features (Topic 8 §2.5, §3.4). A **standalone fallback** `compute_mdx_features.py` re-extracts only the features (no LRP backward) when an older gather produced partials without them (`:1-13`); it forward-passes the backbone and writes the same `mdx_features.npz` schema (`:114-131`). MDX-v2's F_c-256 features are **not** precomputed on HPC — Topic 8 §3.4 notes MDX-v2 runs locally only.

### 2.5 The /ptmp scratch + git-tracked-results transfer model

Viper has two relevant filesystems: `/u/$USER` (home, where the repo lives) and `/ptmp/$USER` (large scratch). The model splits storage accordingly:

- **Raw frames and all intermediate partials live in `/ptmp`** (`/ptmp/$USER/atoms_baseline/`, `atoms_test/`, etc., hardcoded into the SBATCH log paths, e.g. `array_task.sh:13`). Frames are large and gitignored; they are pushed to `/ptmp` by `sync_to_hpc.sh` via an HTTP server + SSH reverse tunnel (Viper is not directly reachable; §3.10).
- **Small numeric outputs travel back through git.** `collect_results.sh` copies the gather outputs from `/ptmp` into the repo's `data/<AGENT>/…` tree and `git add -f`s them (they are gitignored), printing the commit command (it never commits/pushes itself, `:255-263`). Locally the operator `git pull`s and flips the matching `RECOMPUTE_*` flag to `False`. GitHub's 100 MB/file limit is the reason only the small float arrays (profiles, logits, `baseline_*.npz`, `mdx_features.npz`) ride git, never the frames (`cluster_explanations.md:126-127`).

This is why the analysis is reproducible from the repo alone (the profiles are committed) even though the frames are not (finding 5.4, Topic 5: dataset membership is independently non-reproducible).

### 2.6 What stays LOCAL and why

Everything in `run_analysis.py` / `run_online_analysis.py` *except* the LRP+ATOMs profile computation and the MDX feature extraction stays local: the labeled-set build can run locally too (`REAPPLY_PERTURBATIONS`), but more importantly the **detector fitting (Mahalanobis, GMM, MDX, k-NN), scoring, ROC/AUC evaluation, K-sweep, val K-selection, and all figures are local-only** (Topics 7–8). These are cheap (they operate on the small profile arrays, not images), require no model forward pass, and benefit from fast iteration (e.g. `sweep_clusters.py` re-runs K=8/10/12 in seconds because only the GMM changes). Keeping them local avoids the Viper queue latency and the transfer round-trip for every experiment tweak. The HPC layer is run *once per (agent, mode, dataset)*; the local analysis is run *many times*.

### 2.7 No single config shared between local and HPC

The HPC chunk workers do **not** import `atoms_config.py`. Hyperparameters that the local pipeline reads from `conf` are instead **hardcoded into the worker source or passed as CLI defaults in the SBATCH wrappers** — `p_relevance=0.25` and `default_cmd=2` are literals in every chunk worker (`compute_baseline_chunk.py:132-133`); `--mode-analysis` defaults to 1 in the workers (`:37`); PGD ε/steps/target are SBATCH-level env defaults (`array_test_task.sh:54-56`). This is by design — the workers must run in a stub environment without CARLA — but it means the two execution paths can silently diverge (findings 4.3, 1.1, 10.x). There is no assertion or shared constant module coupling them.

---

## 3. Implementation details

### 3.1 The workflow matrix (the central artefact)

Every cell is `(prep, array-compute, gather)` for one workflow × agent. Submit script in **bold**; the array task script and worker `.py` follow.

| Workflow | Agent | prep | array-compute | gather | final output |
|---|---|---|---|---|---|
| **baseline** | TFV6 | *(none — array reads run files directly)* | **submit_baseline.sh** → `array_task.sh` → `compute_baseline_chunk.py --agent TFV6` | `gather_task.sh` → `gather_baseline.py` | `baseline_<mode>.npz` + `mdx_features.npz` |
| baseline | WoR | *(none)* | **submit_baseline_wor.sh** → `array_task_wor.sh` → `compute_baseline_chunk.py --agent WOR` | `gather_task_wor.sh` → `gather_baseline.py` | `baseline_<mode>.npz` (576-d MDX) |
| **test** | TFV6 | `prep_test_task.sh` → `prep_test.py` (5-way 20 %, PGD deferred) | **submit_test.sh** → `array_test_task.sh` → `compute_test_chunk.py` (crafts PGD) | `gather_test_task.sh` → `gather_test.py` | `test_profiles_<mode>.npy` + `test_speed_logits_<mode>.npy` |
| test | WoR | `prep_test_task_wor.sh` → `prep_test_wor.py` (4-way 25 %, **no PGD**) | **submit_test_wor.sh** → `array_test_task_wor.sh` → `compute_test_chunk.py --agent WOR` | `gather_test_task_wor.sh` → `gather_test.py --agent WOR` | `test_profiles_<mode>.npy` + `test_logits_<mode>.npy` |
| **val** | TFV6 | `prep_test_task.sh` → `prep_test.py` (same 5-way spec) | **submit_val.sh** → `array_test_task.sh` (reused) | `gather_test_task.sh` (reused, `SPEED_LOGITS_OUT` overridden) | `val_profiles_<mode>.npy` + `val_speed_logits_<mode>.npy` |
| **live_pert** | TFV6 | `prep_live_pert_task.sh` → `prep_live_pert.py` (concat, wide-only) | **submit_live_pert.sh** → `array_live_pert_task.sh` → `compute_live_pert_chunk.py` (no attack) | `gather_live_pert_task.sh` → `gather_live_pert.py` | `live_pert_profiles_<mode>.npy` per-variant |
| live_pert | WoR | `prep_live_pert_task_wor.sh` → `prep_live_pert_wor.py` (concat, both cams) | **submit_live_pert_wor.sh** → `array_live_pert_task_wor.sh` | `gather_live_pert_task_wor.sh` → `gather_live_pert.py --agent WOR` | `live_pert_profiles_<mode>.npy` + `live_pert_action_logits_<mode>.npy` |

The val workflow is the test workflow with different filenames: `submit_val.sh` reuses `array_test_task.sh` and `gather_test_task.sh`, only renaming the outputs (`val_profiles_*`, `val_speed_logits_*`) by exporting `PROFILES_OUT`/`SPEED_LOGITS_OUT` (`submit_val.sh:44-45,102-106`). The same 5-way labeled spec is applied (it routes through `prep_test_task.sh`).

### 3.2 The baseline chunk worker (`compute_baseline_chunk.py`)

One task = one run file. It builds the LRP wrapper (`build_tfv6_lrp` `:79-109` loads `config.json` + the **first** `model*.pth` via `sorted(...)[0]` dropping shape-mismatched keys, the single-member load of finding 2.5; `build_wor_lrp` `:56-76` loads `config_leaderboard.yaml` + `main_model*.th`), then constructs `ATOMsCarla` with the agent's class map (`TFV6_CLASSES` else `CARLA_CLASSES`) and the **hardcoded** filter:

```
atoms = ATOMsCarla(
    lrp_model     = lrp,
    p_relevance   = 0.9,    # FC_RELEVANCE_FILTER   ← compute_baseline_chunk.py:132 (was 0.25; fixed 2026-06-14, finding 4.3)
    default_cmd   = 2,      # DEFAULT_CMD (FOLLOW_LANE)
    mode_analysis = args.mode_analysis,   # default 1 (:37)
    use_reduced   = False,
    class_map     = class_map,
)
```

The `0.25` literal contradicted `atoms_config.py:24 FC_RELEVANCE_FILTER = 0.9` (finding 4.3 — **fixed 2026-06-14: all three chunk workers now use 0.9; profiles computed before that date used 0.25**); the comment `default_cmd = 2 # (FOLLOW_LANE)` is itself wrong under the 0-based mapping where 2 = STRAIGHT (finding 1.7/2.12). The per-frame loop (`:155-195`) calls `atoms.process_frame(wide, narr, seg_wide, seg_narr, cmd=cmd, spd=spd)` — **`spd` is passed** (unlike the local offline test loop, finding 4.4). Per frame it also collects the MDX feature: TFV6 → `lrp.get_backbone_features(wide)` (512-d) + proxy `[0.0, min(spd/25,1), 1 if spd<0.5 else 0]` (`:186-189`); WoR → `model.get_features(wide, narr)` (576-d) + a true action proxy decoded from the 28-bin joint π(a|s) marginals (`:166-183`). Output partial NPZ holds `series`, `backbone_features`, `mdx_actions`, `class_ids`, `class_names` (`:207-214`).

### 3.3 The test chunk worker and deferred-PGD crafting (`compute_test_chunk.py`)

Same skeleton, plus the **deferred PGD attack** (Topic 6 §2.3). `build_tfv6_lrp` here additionally returns the raw `TFv6` model (`:110`) because the attack needs a full `forward/backward`, which the LRP wrapper cannot supply. The ATOMs filter is again `p_relevance=0.25, default_cmd=2` (`:153-154`). PGD is enabled only for TFV6 when the labeled set carries a `perturbation` field (`:168-169`); `pm.attack_interval = 1` forces a fresh δ per attacked frame (`:174`). For each frame labeled `"pgd"` (`:194`):

- ε is read from the **recorded** `data["intensity"][i]`, falling back to `--pgd-epsilon` if missing/zero (`:195-197`) — so the *recorded* ε (written by prep, default 12.0) takes precedence over the array-task default (14.0). This is the precise mechanism of finding 1.1: prep records 12, the array fallback is 14, and the config claims 14 "must match prep".
- `_make_minimal_data(spd, device, cmd=cmd)` builds the conditioning dict and a zero radar tensor is supplied to keep `forward()` from crashing on the vision-only checkpoint's randomly-initialised radar branch (`:198-209`).
- `pm.pgd_attack_tfv6(nets=[tfv6_model], data=pgd_data, target=args.pgd_target, epsilon=eps, n_steps=args.pgd_steps)` crafts the adversarial RGB against a **single model** (`:210-216`), in contrast to the live recording's 3-model ensemble attack (findings 2.5/6.8).

The crafted RGB replaces `wide` (`:217`), so both the ATOMs profile and the PEOC logits see the attacked pixels. PEOC logits are then collected cheaply (`get_speed_logits` TFV6 / `get_action_logits` WoR, `:223-226`). The partial NPZ stores `profiles`, `chunk_start`, `chunk_end`, `class_ids`, `class_names`, and the logits under key `speed_logits` (TFV6) or `action_logits` (WoR) (`:243-251`).

### 3.4 The live-pert chunk worker (`compute_live_pert_chunk.py`)

Identical to the test worker minus all PGD machinery — the live perturbation (incl. PGD) was already baked into the recorded pixels during driving (Topic 9 §2.4), so there is **no attack crafting** here (`:9-11`). Same `p_relevance=0.25, default_cmd=2` hardcode (`:141-142`), same `--mode-analysis` default 1 (`:45`), `process_frame` called with `spd` (`:165`). Partial NPZ schema matches the test worker. This is the chain documented from the online side in Topic 9 §3.9.

### 3.5 The standalone MDX feature precompute (`compute_mdx_features.py`)

A self-contained script (it inserts the stub + project paths itself, `:40-43`) that globs `run_*.npz`, forward-passes the backbone per frame (`lrp.get_backbone_features`, `:116`), builds the same `[0, min(spd/25,1), brake]` proxy (`:119`), and writes `mdx_features.npz` with `features [N,512]` + `actions [N,3]` (`:131`). It exists for the case where partials predate backbone-feature extraction; it does **no LRP backward** and runs in minutes (its docstring `:12-13`). It is TFV6-only (no `--agent`). Note its `--frames-dir` defaults are absolute `paulkull` paths in the docstring example (`:17-19`), but they are CLI args, not literals in code.

### 3.6 The prep stage — labeled-set / concat construction and meta-file sizing

The prep scripts are Topic 6's labeled-set construction running on a CPU node; here the **HPC role** is the meta file and the index contract.

- **`prep_test.py`** (TFV6): loads all `run_*.npz`, assigns frames to the **5-way 20 % spec** `_SPEC = [None, gaussian_noise, brightness_scale, camera_loss, pgd]` (`:35-42`) via a seed-42 shuffle (`:85-93`), applies the image-space perturbations, and records `pgd` frames with clean pixels + label 1 (`:131-137`). Writes `test_labeled.npz` (with `label`, `perturbation`, `intensity` arrays the array job reads) **and `test_meta.txt`** = total frame count, "so submit_test.sh can size the array job if needed" (`:170-171`). In practice `submit_test.sh` re-counts frames from the labeled file directly (`:44-48`) rather than reading the meta file, so `test_meta.txt` is informational.
- **`prep_test_wor.py`** (WoR): the **4-way 25 % spec with no PGD** (`:26-33`) — a parity break from the TFV6 5-way mix and the WoR *local* 5-way-with-PGD spec (finding 6.5). Preserves both cameras (`:111-134`).
- **`prep_live_pert.py`** / **`prep_live_pert_wor.py`**: pure NumPy concatenation of `run_<pert>_live_pert_*.npz` into `live_pert_concat.npz`, plus **`live_pert_meta.txt`** (total frame count, "read by submit_live_pert.sh to size the SLURM array", `:83-84`). The TFV6 version is wide-only and synthesises a `run_id`; the WoR version preserves `narr_rgb`/`seg_red_narr` and *requires* `narr_rgb` (`prep_live_pert_wor.py:58-62`). Neither carries `is_perturbed` (Topic 9 §3.9: the injection index is reconstructed locally at plot time, not from the concat).

Notably, `submit_live_pert.sh` does **not** read `live_pert_meta.txt`; it hardcodes an upper bound `MAX_FRAMES=200` and sizes the array to `ceil(200/20)=10` tasks per file, relying on the worker's empty-partial-past-end behaviour (`submit_live_pert.sh:42-45`). So the meta-file-sizing claim in the prep docstring is only realised for test/val (and even there, indirectly).

### 3.7 Array sizing formula and SLURM resources

The array size is derived differently per workflow:

| Workflow | `#SBATCH --array` | Formula | Source |
|---|---|---|---|
| baseline (TFV6/WoR) | `0-(N_files-1)` | one task per run file | `submit_baseline.sh:54,65` |
| test / val | `0-(N_TASKS-1)`, `N_TASKS = ceil(N_FRAMES / CHUNK_SIZE)` | frames / chunk | `submit_test.sh:59-60` |
| live_pert | `0-(N_TASKS-1)`, `N_TASKS = ceil(MAX_FRAMES / CHUNK_SIZE)`, `MAX_FRAMES=200` | fixed upper bound, per source file | `submit_live_pert.sh:43-45` |

`CHUNK_SIZE` defaults to **20** frames/task everywhere (`submit_test.sh:35`, `submit_val.sh:39`, `submit_live_pert.sh:39`). The chunk worker computes `CHUNK_START = task_id * CHUNK_SIZE`, `CHUNK_END = CHUNK_START + CHUNK_SIZE` in the array task script (`array_test_task.sh:32-33`). The actual `#SBATCH` resource requests (no GPU anywhere):

| Stage | partition/account | `--cpus-per-task` | `--mem` | `--time` | Source |
|---|---|---|---|---|---|
| baseline array | *(none set — commented `--account` placeholder)* | 4 | 24000MB | 24:00:00 | `array_task.sh:15-19` |
| WoR baseline array | (none) | 4 | 16000MB | 24:00:00 | `array_task_wor.sh:15-18` |
| test array | (none) | 4 | 16000MB | 24:00:00 | `array_test_task.sh:16-19` |
| live_pert array | (none) | 4 | 16000MB | 24:00:00 | `array_live_pert_task.sh:16-19` |
| gather (all) | (none) | 1 | 80000MB | 05:00:00 | `gather_task.sh:13-16` |
| prep_test | (none) | 4 | 32000MB | 03:00:00 | `prep_test_task.sh:14-17` |
| prep_live_pert | (none) | 2 | 16000MB | 01:30:00 | `prep_live_pert_task.sh:16-19` |

**No `--partition` and no active `--account`** are set — every script leaves an `# #SBATCH --account=YOUR_ACCOUNT` *comment* placeholder (`array_task.sh:19-20`), relying on the operator's default allocation. The 24 h array wall-time and the absence of any GPU request confirm the CPU-bound design (finding 10.1).

### 3.8 The SLURM environment and the stubs

Every task script does the same env setup (`array_task.sh:22-28`): `module purge && module load python-waterboa/2025.06`, `source /u/$USER/venvs/pcla/bin/activate`, `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`, and a `PYTHONPATH` ordered **stubs-first**: `$CODE_DIR/hpc/stubs:$CODE_DIR:$CODE_DIR/pcla_agents/transfuserv6`. The stubs-first ordering is essential:

- **`stubs/carla.py`** is a generic mock module (`__getattr__` returns a `_Mock` class for any attribute) that satisfies every `import carla` in the TFV6/WoR codebase without the CARLA simulator installed — the analysis code paths never call real CARLA, so the mocks resolve type annotations and module constants at import time (`stubs/carla.py:1-29`).
- **`stubs/beartype.py`** is a no-op `beartype` decorator + `BeartypeConf`/`BeartypeStrategy`/`roar` shims, "because the TFV6 codebase has invalid string annotations (e.g. `'torch.Union'`) that newer beartype versions reject at import time" (`stubs/beartype.py:1-7`) — the same corrupted-annotation family as finding 2.8. Disabling beartype on HPC sidesteps the import crash and removes runtime type-check overhead.

The venv is built once by **`setup_venv.sh`** (`python -m venv /u/$USER/venvs/pcla`, `pip install -r hpc/requirements_hpc.txt`), which prints the installed torch/zennit versions (`:38-39`). `requirements_hpc.txt` pins `torch==2.6.0+cpu` / `torchvision==0.21.0+cpu` from the CPU wheel index, `timm`, **unpinned `zennit`** (finding 10.2, relates to 3.10 — zennit version determines canonizer semantics), `beartype`, `opencv-python-headless`, `numpy>=2.0`, scipy/sklearn/pandas/matplotlib/pyyaml. The header is internally inconsistent on the Python version (finding 10.3): line 2 says "Tested on Python 3.13", line 8 says "torch 2.5.0 is the earliest build for Py3.13", line 12 then pins **2.6.0** — and the repo's `hpc/__pycache__` is `cpython-310`, matching CLAUDE.md's stated Python 3.10. The HPC venv (3.13) and the local env (3.10) are different interpreters.

### 3.9 Gather ordering and the output contract

All three gather scripts concatenate partials but order them differently:

- **`gather_baseline.py`** sorts partials by the numeric array-task-id suffix `int(f.stem.split("_")[-1])` (`:51-52`), concatenates `series`, computes `mean`/`cov`, and writes `baseline_<mode>.npz` (`series`, `mean`, `cov`, `class_ids`, `class_names`, `cmd_filter=-1`, `n_frames`) plus, when all partials carry them, `mdx_features.npz` (`features`, `actions`) (`:82-114`). If any partial lacks `backbone_features`, MDX output is skipped with a warning (`:76-79`).
- **`gather_test.py`** loads each partial, reads its `chunk_start`, and **sorts by `chunk_start`** before concatenation (`:71`), explicitly "so the final arrays match the original frame ordering in test_labeled.npz" (`:9-10`); empty past-the-end chunks are dropped (`:73-76`). Writes `test_profiles.npy` + the agent-keyed logits file (`test_speed_logits.npy` TFV6 / `test_logits.npy` WoR, `:43-44`).
- **`gather_live_pert.py`** is identical in ordering and writes `live_pert_profiles.npy` + `live_pert_speed_logits.npy` / `live_pert_action_logits.npy` (`:40-41,73-85`).

The gather **task** scripts add the mode suffix at the path level: `gather_test_task.sh:37` names the logits `test_speed_logits_${MODE}.npy` and `submit_test.sh:40` names the profiles `test_profiles_${MODE}.npy`; `gather_task.sh:33` names `baseline_${MODE}.npz`. So on disk every output is mode-suffixed even though the Python script's default filename is not.

**The live-pert naming hazard (finding 9.11, mechanism confirmed):** `submit_live_pert.sh` submits **one chain per source file** into a per-variant subdir `$WORK_DIR/$VARIANT/` (`:69-75`), and the gather writes `live_pert_profiles_${MODE}.npy` into that subdir — **the variant is in the directory path, not the filename**. The local loader (`run_online_analysis.py:412`) expects `live_pert_profiles_{variant}_{mode}.npy` as a *filename*. `collect_results.sh` reconciles this: its `live_pert` branch iterates the per-variant subdirs and **renames on copy** to `live_pert_profiles_{VARIANT}_{MODE}.npy` in the destination (`:161-194`). So the rename the online doc flags as "manual" is actually performed by `collect_results.sh` — but only if `collect_results.sh live_pert` is used; the gather-script's own printed copy instructions (`gather_live_pert.py:90-101`) copy the variant-less `live_pert_profiles.npy`, which would *not* be picked up (finding 10.4).

### 3.10 The operator transfer model (`sync_to_hpc.sh`, `collect_results.sh`) and `cluster_explanations.md`

**Frames up (`sync_to_hpc.sh`):** Viper is unreachable directly, so the script *prints instructions* (it does not transfer itself) for an HTTP-server + SSH-reverse-tunnel: Terminal 1 runs `python -m http.server 8888` in the frames dir; Terminal 2 opens a tunnelled Viper shell (`ssh -R 9999:localhost:8888 -J <gate> <viper>`); on Viper `wget -r -np -nd -A '*.npz' http://localhost:9999/` pulls the frames into `/ptmp` (`sync_to_hpc.sh:48-69`). It handles only `baseline`/`test` frames (`:74-86`) — val and live_pert frames must reuse the same tunnel manually (documented in `cluster_explanations.md:211-216,302-307`).

**Results down (`collect_results.sh`):** runs **on Viper**, copies gather outputs from `/ptmp/$USER/atoms_[wor_]<pipeline>[_alt]` into `data/<AGENT>/…` and `git add -f`s them (`:96-153`). It encodes the full source→dest map including the agent-specific logit names (`test_speed_logits` vs `test_logits`, `live_pert_speed_logits` vs `live_pert_action_logits`, `:42-43`), `find`s files even when nested under `partials/mode_*` (`:198`), supports `--alt` (redirects to `*_data_alt`, `:140`), `--dry-run`, `--no-add`, and exits non-zero on any missing file so a half-finished gather cannot silently stage a partial set (`:281`). It never commits — it prints the commit command (`:255-263`).

**`docs/cluster_explanations.md`** is the human guide for both. It is broadly accurate but has drifted from the scripts in places that matter:
- It states the array crafts PGD "`target=steer_right`, `ε=12`, 10 steps" (`:160-162,199`), but the actual defaults are **`target=brake`** (`array_test_task.sh:54`, `compute_test_chunk.py:47`), **8 steps** (`:53`), and the ε is read from the labeled file first (recorded 12) with array fallback 14 — three mismatches (finding 10.5).
- It calls the array tasks "GPU"/"parallel ATOMs tasks" without noting they are CPU-only (finding 10.1 context).
- `gather_test_task.sh` also runs `visualize_perturb.py` after the profile gather to produce `perturb_samples.png` (`:48-59`), which `cluster_explanations.md` does not mention; that script defaults `--n-cameras 6` while TFV6 frames are 3-camera (finding 2.3), so its centre-camera crop is mis-sliced (finding 10.6).

### 3.11 The /ptmp directory convention

The hardcoded scratch roots, all under `/ptmp/$USER/`:

| Workflow | Frames | Work dir | Partials |
|---|---|---|---|
| baseline (TFV6) | `atoms_baseline/frames` | `atoms_baseline/partials` | `partials/mode_<mode>` |
| baseline (WoR) | `atoms_wor_baseline/frames` | `atoms_wor_baseline/partials` | `partials/mode_<mode>` |
| test | `atoms_test/frames` | `atoms_test` | `atoms_test/partials/mode_<mode>` |
| val | `atoms_val/frames` | `atoms_val` | `atoms_val/partials/mode_<mode>` |
| live_pert | `atoms_live_pert/frames` | `atoms_live_pert/<variant>` | `<variant>/partials/mode_<mode>` |

The `partials/mode_<mode>` nesting (`submit_test.sh:39`) keeps mode-1 and mode-2 partials separate so both modes can be computed without collision, mirroring the local mode-suffixing (Topic 8 §2.2). The username `paulkull` is hardcoded into every example invocation in `cluster_explanations.md` (e.g. `:23,143`) and `compute_mdx_features.py`'s docstring; the scripts themselves use `$USER` (finding 10.7).

---

## 4. Parameters & magic constants

| Constant | Value | Where | Configurable? | Effect |
|---|---|---|---|---|
| `p_relevance` (ATOMs filter) | **0.9** (fixed 2026-06-14; was `0.25`) | `compute_baseline_chunk.py:132`, `compute_test_chunk.py:153`, `compute_live_pert_chunk.py:141` | **hardcoded** | ATOMs mass filter; now matches config `FC_RELEVANCE_FILTER=0.9` (`atoms_config.py:24`). **Profiles computed before 2026-06-14 used 0.25** (finding 4.3) |
| `default_cmd` | 2 | same three workers (e.g. `:133`) | hardcoded | F_c command conditioning; comment "(FOLLOW_LANE)" wrong (2=STRAIGHT, finding 1.7/2.12) |
| `--mode-analysis` default | 1 | all chunk workers `:37/:43/:45`; array tasks `${MODE_ANALYSIS:-1}` (`array_task.sh:48`) | CLI default | mode where `p_relevance` matters; submit scripts default mode 1 (`submit_test.sh:36`) |
| `CHUNK_SIZE` | 20 | `submit_test.sh:35`, `submit_val.sh:39`, `submit_live_pert.sh:39` | CLI arg | frames per array task |
| `MAX_FRAMES` (live array bound) | 200 | `submit_live_pert.sh:43` | **hardcoded** | upper frame bound per live-pert file → 10 tasks; past-end tasks write empty partials |
| array size (baseline) | `N_files` | `submit_baseline.sh:46,65` | derived | one task per run file |
| array size (test/val) | `ceil(N_frames/20)` | `submit_test.sh:59` | derived | frames / chunk |
| `--pgd-epsilon` (prep record) | **12.0** | `prep_test.py:54`, `prep_test_task.sh:37` | CLI/env | ε recorded into test_labeled.npz |
| `--pgd-epsilon` (array fallback) | **14.0** | `compute_test_chunk.py:50`, `array_test_task.sh:55` | CLI/env | fallback ε when recorded intensity 0; ≠ prep 12 (finding 1.1) |
| `--pgd-target` | "brake" | `compute_test_chunk.py:47`, `array_test_task.sh:54` | CLI/env | TFV6 PGD objective (cluster_explanations.md says steer_right — finding 10.5) |
| `--pgd-steps` | 8 | `compute_test_chunk.py:53`, `array_test_task.sh:56` | CLI/env | PGD iterations (cluster_explanations.md says 10 — finding 10.5) |
| `attack_interval` | 1 | `compute_test_chunk.py:174` | hardcoded | fresh δ per attacked frame |
| `pgd nets` | `[tfv6_model]` (single) | `compute_test_chunk.py:211` | hardcoded | single-member attack vs live 3-ensemble (findings 2.5/6.8) |
| TFV6 test spec | 5-way 20 % (incl. pgd) | `prep_test.py:35-42` | code | labeled mix |
| WoR test spec | 4-way 25 % (no pgd) | `prep_test_wor.py:26-33` | code | labeled mix; parity break (finding 6.5) |
| labeling seed | 42 | `prep_test.py:51,86`, `prep_test_task.sh:36` | CLI | frame-to-entry shuffle (must match local seed 42) |
| MDX backbone dim | 512 (TFV6) / 576 (WoR) | `compute_baseline_chunk.py:151,186` | code | MDX-v1 feature space |
| array `--cpus-per-task` | 4 | `array_task.sh:16` | SBATCH | OMP threads per task |
| array `--mem` | 24000MB (baseline) / 16000MB (test/live/WoR) | `array_task.sh:17`; `array_test_task.sh:18` | SBATCH | per-task RAM |
| array `--time` | 24:00:00 | `array_task.sh:18` | SBATCH | wall-time |
| gather `--mem` / `--time` | 80000MB / 05:00:00 | `gather_task.sh:15-16` | SBATCH | concat node |
| `--gpus` | *(none requested)* | all SBATCH | — | CPU-only cluster (finding 10.1) |
| `--account` / `--partition` | unset (comment placeholder) | `array_task.sh:19-20` | SBATCH | relies on default allocation |
| venv path | `/u/$USER/venvs/pcla` | `setup_venv.sh:13`, all task scripts `:24` | hardcoded | activated per task |
| module | `python-waterboa/2025.06` | all task scripts `:23` | hardcoded | Viper Python module |
| torch / torchvision | 2.6.0+cpu / 0.21.0+cpu | `requirements_hpc.txt:14-15` | pinned | CPU wheels |
| zennit | *(unpinned)* | `requirements_hpc.txt:19` | unpinned | canonizer semantics undefined (finding 10.2; relates 3.10) |
| Python (HPC) | 3.13 (header) vs 3.10 (`__pycache__`) | `requirements_hpc.txt:2,8`; `hpc/__pycache__` | — | version inconsistency (finding 10.3) |
| `/ptmp` roots | `atoms_[wor_]<pipeline>` | SBATCH log paths, `collect_results.sh:99` | hardcoded | scratch layout |
| HTTP / tunnel ports | 8888 / 9999 | `sync_to_hpc.sh:30-31` | code | transfer ports |
| HPC username (docs) | `paulkull` | `cluster_explanations.md` throughout; `compute_mdx_features.py:17-19` docstring | hardcoded | non-portable examples (finding 10.7) |
| `visualize_perturb --n-cameras` | 6 | `visualize_perturb.py:35` | CLI default | wrong for 3-cam TFV6 frames (finding 10.6) |

---

## 5. Known limitations & open issues

- **The cluster is CPU-only, not GPU** (finding 10.1) — `requirements_hpc.txt:1` says "MPCDF Viper-CPU", pins CPU torch wheels, and **no SLURM script requests `--gpus`**; the workers' `cuda if available` branch always falls to CPU on Viper. `cluster_explanations.md` and the project framing call the array stage "GPU" — it is not. The 24 h array wall-times reflect CPU LRP cost.
- **Chunk workers hardcoded `p_relevance=0.25` ≠ config 0.9** (finding 4.3) — **FIXED 2026-06-14**: `compute_baseline_chunk.py:132`, `compute_test_chunk.py:153`, `compute_live_pert_chunk.py:141` now all use `0.9`, matching `atoms_config.py:24`. ⚠ **Every HPC mode-1 profile computed before 2026-06-14 used a 25 % mass filter** — those committed profiles must be recomputed to obtain 0.9 results, or 0.25 must be reported as the value actually used. Long-term fix (still open): source `p_relevance` from a single shared config rather than hardcoding (finding 10.8).
- **PGD ε: prep records 12, array fallback is 14, config claims 14** (finding 1.1, confirmed) — `prep_test.py:54`/`prep_test_task.sh:37` default 12.0; `compute_test_chunk.py:50`/`array_test_task.sh:55` default 14.0; `atoms_config.py:84` is 14.0 with comment "must match hpc/prep_test.py default". Because the worker reads the recorded intensity first (`compute_test_chunk.py:195-197`), the *effective* attack ε is 12 unless the prep default is overridden — so the offline label ε and the config-claimed ε disagree.
- **WoR HPC test prep is 4-way no-PGD; TFV6 is 5-way with-PGD** (finding 6.5) — `prep_test_wor.py:26-33` vs `prep_test.py:35-42`. A WoR labeled set built on HPC has a different composition than the WoR *local* 5-way spec and than the TFV6 HPC set, breaking the local↔HPC parity the TFV6 path relies on.
- **Single-member attack/analysis vs live ensemble** (findings 2.5/6.8) — `compute_test_chunk.py:211` attacks `nets=[tfv6_model]` and every worker loads `sorted(...)[0]` only; the live recordings were driven and PGD-attacked against the 3-model ensemble. HPC profiles explain a different policy than the one that produced the recordings.
- **Live-pert gather naming requires `collect_results.sh` to rename** (findings 9.11, 10.4) — gather writes `live_pert_profiles_<mode>.npy` with the variant in the *directory*, not the filename; the local loader wants `live_pert_profiles_{variant}_{mode}.npy`. `collect_results.sh:161-194` renames on copy, but `gather_live_pert.py`'s own printed copy instructions (`:90-101`) copy the variant-less name, which the loader will not pick up. The two reconciliation paths disagree.
- **`cluster_explanations.md` is stale on the PGD attack** (finding 10.5) — it documents `target=steer_right, ε=12, 10 steps` (`:160-162,199`); actual defaults are `target=brake`, ε read-from-file (recorded 12 / fallback 14), 8 steps. It also omits the CPU nature of the array stage and the `visualize_perturb.py` step run by `gather_test_task.sh`.
- **`visualize_perturb.py` defaults to 6 cameras for 3-cam TFV6 frames** (finding 10.6) — `:35` `--n-cameras 6`; TFV6 LEAD frames are 3-camera 1152 px (finding 2.3), so the centre-camera crop (`:71` `n_cameras // 2 = 3`) indexes a non-existent camera band. The QC figure shows the wrong slice.
- **`zennit` unpinned in `requirements_hpc.txt`** (finding 10.2; relates 3.10) — the installed zennit version determines the canonizer/composite semantics relied on by findings 3.3/3.4. An unpinned HPC install can produce subtly different attributions than the local env.
- **Python version inconsistency** (finding 10.3) — `requirements_hpc.txt:2,8` says "Python 3.13 / torch 2.5.0 earliest", line 12 pins torch 2.6.0, and `hpc/__pycache__` is `cpython-310` (matching CLAUDE.md's Python 3.10). The HPC venv and local env are different interpreters; the header's own claims contradict each other.
- **No shared config between local and HPC** (finding 10.8) — the chunk workers do not import `atoms_config.py`; every shared hyperparameter (p_relevance, default_cmd, mode, PGD ε/steps/target, seed) is duplicated as a literal or CLI default in the worker/SBATCH layer with no coupling or assertion. This is the root cause of findings 4.3, 1.1, 6.5.
- **Hardcoded usernames/paths in docs** (finding 10.7) — `cluster_explanations.md` and `compute_mdx_features.py`'s docstring hardcode `paulkull` and absolute `/u/paulkull` / `C:\Users\paulk` paths throughout the examples; the scripts use `$USER` but the operator guide is non-portable.
- **`live_pert_meta.txt` written but not used for array sizing** (finding 10.9, low) — `prep_live_pert.py:83-84` writes the meta file "to size the SLURM array", but `submit_live_pert.sh:43` uses a hardcoded `MAX_FRAMES=200` instead. `test_meta.txt` is likewise written but `submit_test.sh` re-counts from the labeled file. The meta files are informational only.

---

## 6. Cross-references

- **01_architecture_overview.md** — `atoms_config.py` as the local single source of truth that the HPC workers deliberately do *not* import (finding 10.8); `MODE_ANALYSIS`, `FC_RELEVANCE_FILTER`, `DEFAULT_CMD`, `PGD_EPSILON`/`EPSILON`/`PGD_TARGET`/`PGD_N_STEPS`, `RANDOM_SEED`, `EXPERIMENT_VARIANT` (`--alt` in `collect_results.sh`); findings 1.1 (PGD ε), 1.7 (DEFAULT_CMD).
- **02_agents.md** — the single-member `sorted(...)[0]` load in every chunk worker (finding 2.5); the feature/logit extractors HPC calls: TFV6 `get_backbone_features`/`get_speed_logits`, WoR `get_features`/`get_action_logits`; the 512-d vs 576-d MDX feature spaces; the corrupted-annotation family that motivates the beartype stub (finding 2.8); the 3-cam-vs-6-cam frame geometry (finding 2.3 → 10.6).
- **03_lrp.md** — the LRP backward the chunk workers run per frame (`LRPTFv6Model` / `LRPCameraModel`); zennit being unpinned on HPC (finding 10.2; the canonizer semantics of 3.3/3.4/3.10 depend on the installed version).
- **04_atoms.md** — `ATOMsCarla.process_frame` (with `spd`) is the per-frame call in all three workers; the `p_relevance=0.25`-vs-0.9 hardcode (finding 4.3); MODE_ANALYSIS 1 vs 2; the worker imports `TFV6_CLASSES`/`CARLA_CLASSES` from `visualization_carla` (finding 4.9, divergence risk vs the `atoms_carla` copy).
- **05_dataset_creation.md** — the `run_*.npz` frame schema the workers/prep consume; the dataset-membership non-reproducibility (finding 5.4) that the git-tracked-profiles model (§2.5) partially mitigates; `--alt` paths.
- **06_perturbations.md** — `prep_test*.py` is Topic 6's labeled-set construction; the deferred-PGD design crafted on `compute_test_chunk.py` (§3.3); the 5-way TFV6 vs 4-way WoR HPC spec (finding 6.5); single-member vs ensemble attack (finding 6.8); ε divergences (1.1, 6.4).
- **07_distances_and_detectors.md** — `mdx_features.npz` feeds MDX-v1 fit (§2.4); the PEOC logit arrays the workers save feed `ActionEntropyDetector`; the profiles feed every distance detector.
- **08_offline_analysis.md** — the local consumer: `run_analysis.py` STEP 2/3/9 load the HPC `baseline_<mode>.npz` / `mdx_features.npz` / `test_profiles_<mode>.npy`; the `RECOMPUTE_*=False` gates (§2.1 there) that make HPC outputs authoritative; the deferred-PGD load guard (`:840-865`) that merges HPC PGD profiles; the labeling seed 42 parity.
- **09_online_analysis.md** — the live-pert chain documented from the online side (§3.9 there); the HPC↔local profile-naming mismatch (finding 9.11, mechanism confirmed here §3.9); `run_online_analysis.py` consumes the gathered live-pert profiles when `RECOMPUTE_TEST_ATOMS=False`.
- **11_validation_and_testing.md** — the LRP/ATOMs correctness suites that validate the profiles this layer produces.
- **12_visualization.md** — `visualize_perturb.py` (the only figure produced inside the HPC chain, finding 10.6) and the local figures rendered from HPC-computed profiles.
- **99_bugs_and_findings.md** — Topic 10 findings 10.1–10.9; cross-references 1.1, 1.7, 2.3, 2.5, 2.8, 3.10, 4.3, 4.9, 5.4, 6.4, 6.5, 6.8, 9.11.
