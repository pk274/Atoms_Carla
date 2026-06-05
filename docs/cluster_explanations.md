# HPC / Viper Cluster — How-To

Practical reference for transferring data and running jobs on MPCDF Viper.

---

## Network requirements

Viper is not reachable from off-campus without the **MPCDF VPN**.  
Connect to the VPN first, then all SSH/tunnel commands below will work.  
Gate server: `gate1.mpcdf.mpg.de` — only reachable once VPN is active.

---

## Transferring large files to Viper (HTTP reverse tunnel)

Direct rsync/scp to Viper does not work from Windows — you must go through the gateway.  
The cleanest solution requires no intermediate storage on the gateway: serve files from your local machine over HTTP, then pull them down on Viper through an SSH reverse tunnel.

**Make sure the MPCDF VPN is connected before starting.**

### Step 1 — Local Terminal 1: start HTTP server in the directory you want to upload

```powershell
cd data\TFV6\test_data\frames       # or whichever directory
python -m http.server 8888
```

Leave this terminal open for the duration of the transfer.

### Step 2 — Local Terminal 2: open a Viper shell with the reverse tunnel

Run this **on your local machine** (not on Viper):

```powershell
ssh -o 'MACs=hmac-sha2-256-etm@openssh.com' -R 9999:localhost:8888 -J paulkull@gate1.mpcdf.mpg.de paulkull@viper.mpcdf.mpg.de
```

Enter your password + OTP when prompted. This opens an interactive shell **on Viper** with port 9999 tunnelled back to your laptop's port 8888.

> **Common mistake:** do not run this command inside an existing Viper session —
> it must be run from your local machine.

### Step 3 — On Viper (in Terminal 2): download the files

```bash
mkdir -p /ptmp/paulkull/atoms_test/frames
cd /ptmp/paulkull/atoms_test/frames
wget -r -np -nd -A "*.npz" http://localhost:9999/
```

When wget finishes, Ctrl-C the http.server in Terminal 1.

### To upload a different directory

Change the `cd` path in Terminal 1 and the `mkdir`/`cd` path on Viper.  
Ports 8888 / 9999 can be reused each time.

---

## Transferring small results files back (git)

Computed outputs (`baseline_1.npz`, `test_profiles_1.npy`, etc.) are gitignored, so they
must be force-added. **Don't do the `cp` + `git add -f` by hand** — use the helper:

### `hpc/collect_results.sh` (recommended)

One command finds the gather outputs in `/ptmp`, copies them into the right `data/<AGENT>/…`
folder, and `git add -f`s them. It **does not commit or push** — it prints the exact commit
command for you to run.

```bash
# bash hpc/collect_results.sh <pipeline> <agent> <mode> [pert] [options]
#   pipeline : baseline | test | live_pert
#   agent    : tfv6 | wor
#   mode     : 1 | 2
#   pert     : required for live_pert (e.g. pgd)

cd /u/$USER/pcla
bash hpc/collect_results.sh test tfv6 1          # stage TFV6 test profiles+logits, mode 1
bash hpc/collect_results.sh test tfv6 2          # …and mode 2
git commit -m "add TFV6 test results from HPC"   # (the script prints this line for you)
git push
```

It knows every source→destination mapping (incl. the `test_speed_logits` vs `test_logits`
vs `live_pert_action_logits` naming differences) and locates files even when they are nested
under `partials/mode_*`. Options: `--work-dir DIR` (override the default `/ptmp/$USER/atoms_[wor_]<pipeline>`),
`--code-dir DIR` (repo root), `--no-add` (copy only), `--dry-run` (preview). Missing files are
reported and exit non-zero, so a half-finished gather won't silently stage a partial set.

**Locally afterwards:** `git pull`, then set the matching `RECOMPUTE_*` flag to `False`
(`RECOMPUTE_BASELINE`/`RECOMPUTE_MDX_BASELINE` for baseline; `RECOMPUTE_TEST_ATOMS` for test/live).

### Manual fallback

```bash
cp /ptmp/$USER/atoms_baseline/partials/baseline_1.npz /u/$USER/pcla/data/TFV6/baseline_data/baseline_1.npz
cd /u/$USER/pcla && git add -f data/TFV6/baseline_data/baseline_1.npz && git commit -m "…" && git push
```

GitHub has a 100 MB per-file hard limit — only use git for computed outputs
(small float arrays), never for raw frame files.

---

## Full test-set pipeline

### 1. Upload test frames (see above)

Source: `data/TFV6/test_data/frames/`  
Destination on Viper: `/ptmp/paulkull/atoms_test/frames/`

### 2. Submit all jobs in one command (on Viper)

Pass the desired `MODE_ANALYSIS` (1 or 2) as the last argument. Run once per mode to produce both:

```bash
cd /u/paulkull/pcla
git pull
bash hpc/submit_test.sh \
    /ptmp/paulkull/atoms_test/frames \
    /ptmp/paulkull/atoms_test \
    /u/paulkull/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34 \
    "" "" 1   # MODE_ANALYSIS=1  (args 4=CODE_DIR, 5=CHUNK_SIZE left as defaults)
bash hpc/submit_test.sh ... 2   # MODE_ANALYSIS=2
```

This chains three SLURM jobs automatically:
1. **prep** — applies the image-space perturbations → `test_labeled.npz`. The mix is
   a 5-way 20 % split: clean / gaussian_noise / brightness_scale / camera_loss / **pgd**.
   `pgd` frames are *recorded* with clean pixels (prep is model-free); the attack is
   crafted in the array job.
2. **array** — parallel ATOMs tasks (20 frames each), each also computing 8-bin speed
   logits for PEOC. For frames labelled `pgd`, `compute_test_chunk.py` crafts the TFV6
   adversarial image via `pgd_attack_tfv6` (a minimal-data `TFv6.forward` backward pass,
   `target=steer_right`, `ε=12`, 10 steps — override with `PGD_TARGET`/`PGD_EPSILON`/
   `PGD_STEPS`) before running LRP + ATOMs, so both the profile and the PEOC logits see
   the attacked pixels.
3. **gather** — concatenates results → `test_profiles_{MODE}.npy` + `test_speed_logits_{MODE}.npy`

Monitor: `squeue -u paulkull`

### 3. Collect results (on Viper, then git pull locally)

```bash
cd /u/$USER/pcla
bash hpc/collect_results.sh test tfv6 1
bash hpc/collect_results.sh test tfv6 2     # if both modes were computed
git commit -m "add TFV6 test results from HPC"
git push
```

Then locally: `git pull`, set `RECOMPUTE_TEST_ATOMS = False` in `atoms_config.py`, and set `MODE_ANALYSIS` to whichever mode you want to analyse.

`test_speed_logits_{MODE}.npy` is automatically used by `run_analysis.py` for PEOC scoring — no config flag needed.

### Re-running the TFV6 test set with PGD

The 5-way mix (incl. `pgd`) is new. `submit_test.sh` **skips prep if `test_labeled.npz` already
exists**, so to pick up the PGD frames you must delete the stale labelled set and partials first:

```bash
rm -f  /ptmp/$USER/atoms_test/test_labeled.npz      # force prep to rebuild with the 5-way PGD mix
rm -rf /ptmp/$USER/atoms_test/partials              # old profiles are stale (mix changed)
bash hpc/submit_test.sh \
    /ptmp/$USER/atoms_test/frames \
    /ptmp/$USER/atoms_test \
    /u/$USER/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34 \
    "" "" 1                                          # then repeat with trailing 2 for mode 2
# after both gathers finish:
bash hpc/collect_results.sh test tfv6 1 && bash hpc/collect_results.sh test tfv6 2
```

Override the attack with `PGD_TARGET` / `PGD_EPSILON` / `PGD_STEPS` (defaults `steer_right` /
`12` / `10`), e.g. `PGD_EPSILON=8 bash hpc/submit_test.sh …`. The PGD fraction (20 %) shrinks the
other perturbations from 25 %→20 %, so their AUCs will shift slightly on re-run — expected.

---

## Full baseline pipeline

### 1. Upload baseline frames

Source: `data/TFV6/baseline_data/frames/`  
Destination on Viper: `/ptmp/paulkull/atoms_baseline/frames/`

Use the same HTTP tunnel method, pointing Terminal 1 at `data\TFV6\baseline_data\frames`.

### 2. Submit (on Viper)

Pass `MODE_ANALYSIS` (1 or 2) as the 5th argument. Run once per mode:

```bash
bash hpc/submit_baseline.sh \
    /ptmp/paulkull/atoms_baseline/frames \
    /ptmp/paulkull/atoms_baseline/partials \
    /u/paulkull/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34 \
    "" 1   # 4th arg=CODE_DIR (default), 5th=MODE_ANALYSIS
bash hpc/submit_baseline.sh ... "" 2   # mode 2
```

Each array task extracts ATOMs profiles + 512-dim backbone features.  
The gather step writes `baseline_{MODE}.npz` (in `partials/`) and `mdx_features.npz`.

### 3. Collect results

```bash
cd /u/$USER/pcla
bash hpc/collect_results.sh baseline tfv6 1
bash hpc/collect_results.sh baseline tfv6 2     # mdx_features.npz is shared; re-copying is harmless
git commit -m "add TFV6 baseline results from HPC"
git push
```

Then locally: `git pull`, set `RECOMPUTE_BASELINE = False` in `atoms_config.py`, and run `run_analysis.py` once with `RECOMPUTE_MDX_BASELINE = True` — it will detect `mdx_features.npz`, fit MDX in seconds, and save `mdx_parameters.pkl`. After that set `RECOMPUTE_MDX_BASELINE = False` too.

---

## Full live-perturbation pipeline

Live-perturbation data is recorded in CARLA with `LIVE_PERTURBATION_RECORDING_MODE = True`.
The frames are already perturbed at collection time, so no offline perturbation step is
needed — the HPC pipeline is just concatenation + ATOMs.

### 1. Upload live-pert frames

Source: `data/TFV6/test_data/live_pert_frames/`  
Destination on Viper: `/ptmp/paulkull/atoms_live_pert/frames/`

Use the same HTTP tunnel method, pointing Terminal 1 at `data\TFV6\test_data\live_pert_frames`.

### 2. Submit all jobs in one command (on Viper)

Replace `pgd` with whichever perturbation name was used during recording.

```bash
cd /u/paulkull/pcla
git pull
bash hpc/submit_live_pert.sh \
    /ptmp/paulkull/atoms_live_pert/frames \
    /ptmp/paulkull/atoms_live_pert \
    /u/paulkull/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34 \
    pgd "" "" 1   # args 5=CODE_DIR, 6=CHUNK_SIZE (defaults), 7=MODE_ANALYSIS
bash hpc/submit_live_pert.sh ... pgd "" "" 2   # mode 2
```

This chains three SLURM jobs automatically:
1. **prep** — concatenates `run_pgd_live_pert_*.npz` files → `live_pert_concat.npz`
2. **array** — parallel ATOMs tasks (20 frames each), also computing speed logits
3. **gather** — concatenates results → `live_pert_profiles_{MODE}.npy` + `live_pert_speed_logits_{MODE}.npy`

Monitor: `squeue -u paulkull`

### 3. Collect results (on Viper, then git pull locally)

```bash
cd /u/$USER/pcla
bash hpc/collect_results.sh live_pert tfv6 1 pgd
bash hpc/collect_results.sh live_pert tfv6 2 pgd     # mode 2
git commit -m "add TFV6 live_pert pgd results from HPC"
git push
```

Replace `pgd` with whichever perturbation was recorded. Then locally: `git pull`, set `RECOMPUTE_TEST_ATOMS = False` and `MODE_ANALYSIS` to the desired mode in `atoms_config.py`.

---

## WoR HPC pipelines

WoR uses the same SLURM infrastructure as TFV6, but with dedicated scripts that handle
the narrow camera (`narr_rgb`) and the 28-dim joint action logits for PEOC.

### Key differences from TFV6

| Aspect | TFV6 | WoR |
|--------|-------|-----|
| Model dir | `pcla_agents/transfuserv6_pretrained/visiononly_resnet34` | `pcla_agents/wor_pretrained/leaderboard_weights` |
| Backbone features | 512-dim (ResNet34 GAP) | 576-dim (512 wide + 64 narr bottleneck) |
| PEOC logits saved | `speed_logits` [N,8] → `test_speed_logits_{MODE}.npy` | `action_logits` [N,28] → `test_logits_{MODE}.npy` |
| Segmentation classes | TFV6_CLASSES (10) | CARLA_CLASSES (29) |
| Prep script | `prep_test.py` (wide only) | `prep_test_wor.py` (wide + narr) |
| Live-pert prep | `prep_live_pert.py` (wide only) | `prep_live_pert_wor.py` (wide + narr) |
| Submit scripts | `submit_baseline.sh`, `submit_test.sh`, `submit_live_pert.sh` | `submit_baseline_wor.sh`, `submit_test_wor.sh`, `submit_live_pert_wor.sh` |

### One-time setup on Viper

Transfer WoR weights (do this once before any WoR HPC run):

```bash
# Local Terminal 1
cd pcla_agents\wor_pretrained\leaderboard_weights
python -m http.server 8888

# Local Terminal 2
ssh -o 'MACs=hmac-sha2-256-etm@openssh.com' -R 9999:localhost:8888 -J paulkull@gate1.mpcdf.mpg.de paulkull@viper.mpcdf.mpg.de

# On Viper
mkdir -p /u/paulkull/pcla/pcla_agents/wor_pretrained/leaderboard_weights
cd /u/paulkull/pcla/pcla_agents/wor_pretrained/leaderboard_weights
wget http://localhost:9999/config_leaderboard.yaml
wget http://localhost:9999/main_model_10.th
```

### WoR baseline pipeline

#### 1. Upload baseline frames

Source: `data/WOR/baseline_data/frames/`  
Destination on Viper: `/ptmp/paulkull/atoms_wor_baseline/frames/`

Use the same HTTP tunnel method, pointing Terminal 1 at `data\WOR\baseline_data\frames`.

#### 2. Submit (on Viper)

Pass `MODE_ANALYSIS` (1 or 2) as the 5th argument. Run once per mode:

```bash
cd /u/paulkull/pcla
git pull
bash hpc/submit_baseline_wor.sh \
    /ptmp/paulkull/atoms_wor_baseline/frames \
    /ptmp/paulkull/atoms_wor_baseline/partials \
    /u/paulkull/pcla/pcla_agents/wor_pretrained/leaderboard_weights \
    "" 1   # 4th=CODE_DIR (default), 5th=MODE_ANALYSIS
bash hpc/submit_baseline_wor.sh ... "" 2   # mode 2
```

Chains two SLURM jobs automatically:
1. **array** — one task per run file; computes ATOMs profiles + backbone features + MDX actions
2. **gather** — concatenates results → `baseline_{MODE}.npz` (in `partials/`) + `mdx_features.npz`

#### 3. Collect results

```bash
cd /u/$USER/pcla
bash hpc/collect_results.sh baseline wor 1
bash hpc/collect_results.sh baseline wor 2
git commit -m "add WOR baseline results from HPC"
git push
```

Then locally: `git pull`, set `RECOMPUTE_BASELINE = False` and `RECOMPUTE_MDX_BASELINE = False`.
`run_analysis.py` will detect `mdx_features.npz` and fit MDX locally in seconds.

---

### WoR test-set pipeline

#### 1. Upload test frames

Source: `data/WOR/test_data/frames/`  
Destination on Viper: `/ptmp/paulkull/atoms_wor_test/frames/`

#### 2. Submit (on Viper)

```bash
cd /u/paulkull/pcla
git pull
bash hpc/submit_test_wor.sh \
    /ptmp/paulkull/atoms_wor_test/frames \
    /ptmp/paulkull/atoms_wor_test \
    /u/paulkull/pcla/pcla_agents/wor_pretrained/leaderboard_weights \
    "" "" 1   # 4th=CODE_DIR, 5th=CHUNK_SIZE (defaults), 6th=MODE_ANALYSIS
bash hpc/submit_test_wor.sh ... "" "" 2   # mode 2
```

Chains three SLURM jobs automatically:
1. **prep** — applies perturbations to both cameras → `test_labeled.npz`
2. **array** — 20-frame chunks; computes ATOMs profiles + 28-dim PEOC action logits
3. **gather** — concatenates → `test_profiles_{MODE}.npy` + `test_logits_{MODE}.npy`

Monitor: `squeue -u paulkull`

#### 3. Collect results

```bash
cd /u/$USER/pcla
bash hpc/collect_results.sh test wor 1
bash hpc/collect_results.sh test wor 2
git commit -m "add WOR test results from HPC"
git push
```

Then locally: `git pull`, set `RECOMPUTE_TEST_ATOMS = False` and `MODE_ANALYSIS` as desired.

---

### WoR live-perturbation pipeline

Live-perturbation data is recorded in CARLA with `LIVE_PERTURBATION_RECORDING_MODE = True`.
The frames are already perturbed at collection time. WoR-specific: `prep_live_pert_wor.py`
preserves both cameras (unlike the TFV6 version which drops `narr_rgb`).

#### 1. Upload live-pert frames

Source: `data/WOR/test_data/live_pert_frames/`  
Destination on Viper: `/ptmp/paulkull/atoms_wor_live_pert/frames/`

Use the same HTTP tunnel method, pointing Terminal 1 at `data\WOR\test_data\live_pert_frames`.

#### 2. Submit (on Viper)

Replace `pgd` with whichever perturbation name was used during recording.

```bash
cd /u/paulkull/pcla
git pull
bash hpc/submit_live_pert_wor.sh \
    /ptmp/paulkull/atoms_wor_live_pert/frames \
    /ptmp/paulkull/atoms_wor_live_pert \
    /u/paulkull/pcla/pcla_agents/wor_pretrained/leaderboard_weights \
    pgd "" "" 1   # 5th=CODE_DIR, 6th=CHUNK_SIZE (defaults), 7th=MODE_ANALYSIS
bash hpc/submit_live_pert_wor.sh ... pgd "" "" 2   # mode 2
```

Chains three SLURM jobs automatically:
1. **prep** — concatenates `run_pgd_live_pert_*.npz` files, preserving both cameras → `live_pert_concat.npz`
2. **array** — parallel ATOMs tasks (20 frames each), also computing 28-dim PEOC action logits
3. **gather** — concatenates results → `live_pert_profiles_{MODE}.npy` + `live_pert_action_logits_{MODE}.npy`

Monitor: `squeue -u paulkull`

#### 3. Collect results

```bash
cd /u/$USER/pcla
bash hpc/collect_results.sh live_pert wor 1 pgd
bash hpc/collect_results.sh live_pert wor 2 pgd
git commit -m "add WOR live_pert pgd results from HPC"
git push
```

Replace `pgd` with whichever perturbation was recorded. Then locally: `git pull`, set `RECOMPUTE_TEST_ATOMS = False` and `MODE_ANALYSIS` as desired.
