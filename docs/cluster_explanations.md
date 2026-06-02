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

For small computed outputs (`baseline.npz`, `test_profiles.npy`, etc.) the easiest
method is git force-add, since these files are gitignored by default.

**On Viper:**

```bash
# Copy result from ptmp to the git repo
cp /ptmp/paulkull/atoms_baseline/partials/baseline.npz \
   /u/paulkull/pcla/data/TFV6/baseline_data/baseline.npz

cd /u/paulkull/pcla
git add -f data/TFV6/baseline_data/baseline.npz
git commit -m "add TFV6 baseline.npz from HPC"
git push
```

**Locally:**

```bash
git pull
```

GitHub has a 100 MB per-file hard limit — only use this for computed outputs
(small float arrays), never for raw frame files.

---

## Full test-set pipeline

### 1. Upload test frames (see above)

Source: `data/TFV6/test_data/frames/`  
Destination on Viper: `/ptmp/paulkull/atoms_test/frames/`

### 2. Submit all jobs in one command (on Viper)

```bash
cd /u/paulkull/pcla
git pull
bash hpc/submit_test.sh \
    /ptmp/paulkull/atoms_test/frames \
    /ptmp/paulkull/atoms_test \
    /u/paulkull/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34
```

This chains three SLURM jobs automatically:
1. **prep** — applies perturbations → `test_labeled.npz`
2. **array** — 10 parallel ATOMs tasks (20 frames each), each also computing 8-bin speed logits for PEOC
3. **gather** — concatenates results → `test_profiles.npy` + `test_speed_logits.npy`

Monitor: `squeue -u paulkull`

### 3. Download results (on Viper, then git pull locally)

```bash
cp /ptmp/paulkull/atoms_test/test_profiles.npy \
   /u/paulkull/pcla/data/TFV6/test_data/attention/test_profiles.npy
cp /ptmp/paulkull/atoms_test/test_speed_logits.npy \
   /u/paulkull/pcla/data/TFV6/test_data/attention/test_speed_logits.npy

cd /u/paulkull/pcla
git add -f data/TFV6/test_data/attention/test_profiles.npy
git add -f data/TFV6/test_data/attention/test_speed_logits.npy
git commit -m "add TFV6 test_profiles.npy and test_speed_logits.npy from HPC"
git push
```

Then locally: `git pull`, and set `RECOMPUTE_TEST_ATOMS = False` in `atoms_config.py`.

`test_speed_logits.npy` is automatically used by `run_analysis.py` for PEOC scoring — no config flag needed.

---

## Full baseline pipeline

### 1. Upload baseline frames

Source: `data/TFV6/baseline_data/frames/`  
Destination on Viper: `/ptmp/paulkull/atoms_baseline/frames/`

Use the same HTTP tunnel method, pointing Terminal 1 at `data\TFV6\baseline_data\frames`.

### 2. Submit (on Viper)

```bash
bash hpc/submit_baseline.sh \
    /ptmp/paulkull/atoms_baseline/frames \
    /ptmp/paulkull/atoms_baseline/partials \
    /u/paulkull/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34
```

Each array task now also extracts 512-dim backbone features alongside ATOMs profiles.  
The gather step writes both `baseline.npz` and `mdx_features.npz`.

### 3. Download results

```bash
cp /ptmp/paulkull/atoms_baseline/partials/baseline.npz \
   /u/paulkull/pcla/data/TFV6/baseline_data/baseline.npz
cp /ptmp/paulkull/atoms_baseline/partials/mdx_features.npz \
   /u/paulkull/pcla/data/TFV6/baseline_data/mdx_features.npz

cd /u/paulkull/pcla
git add -f data/TFV6/baseline_data/baseline.npz
git add -f data/TFV6/baseline_data/mdx_features.npz
git commit -m "add TFV6 baseline.npz and mdx_features.npz from HPC"
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
    pgd
```

This chains three SLURM jobs automatically:
1. **prep** — concatenates `run_pgd_live_pert_*.npz` files → `live_pert_concat.npz`
2. **array** — 10 parallel ATOMs tasks (20 frames each), also computing speed logits
3. **gather** — concatenates results → `live_pert_profiles.npy` + `live_pert_speed_logits.npy`

Monitor: `squeue -u paulkull`

### 3. Download results (on Viper, then git pull locally)

```bash
PERT=pgd
ATT=/u/paulkull/pcla/data/TFV6/test_data/attention/live_pert/$PERT
mkdir -p $ATT

cp /ptmp/paulkull/atoms_live_pert/live_pert_profiles.npy      $ATT/live_pert_profiles.npy
cp /ptmp/paulkull/atoms_live_pert/live_pert_speed_logits.npy  $ATT/live_pert_speed_logits.npy

cd /u/paulkull/pcla
git add -f data/TFV6/test_data/attention/live_pert/$PERT/live_pert_profiles.npy
git add -f data/TFV6/test_data/attention/live_pert/$PERT/live_pert_speed_logits.npy
git commit -m "add live_pert_profiles for $PERT from HPC"
git push
```

Then locally: `git pull`, and set `RECOMPUTE_TEST_ATOMS = False` in `atoms_config.py`.

---

## WoR HPC pipelines

WoR uses the same SLURM infrastructure as TFV6, but with dedicated scripts that handle
the narrow camera (`narr_rgb`) and the 28-dim joint action logits for PEOC.

### Key differences from TFV6

| Aspect | TFV6 | WoR |
|--------|-------|-----|
| Model dir | `pcla_agents/transfuserv6_pretrained/visiononly_resnet34` | `pcla_agents/wor_pretrained/leaderboard_weights` |
| Backbone features | 512-dim (ResNet34 GAP) | 576-dim (512 wide + 64 narr bottleneck) |
| PEOC logits saved | `speed_logits` [N,8] → `test_speed_logits.npy` | `action_logits` [N,28] → `test_logits.npy` |
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

```bash
cd /u/paulkull/pcla
git pull
bash hpc/submit_baseline_wor.sh \
    /ptmp/paulkull/atoms_wor_baseline/frames \
    /ptmp/paulkull/atoms_wor_baseline/partials \
    /u/paulkull/pcla/pcla_agents/wor_pretrained/leaderboard_weights
```

Chains two SLURM jobs automatically:
1. **array** — one task per run file; computes ATOMs profiles + backbone features + MDX actions
2. **gather** — concatenates results → `baseline.npz` + `mdx_features.npz`

#### 3. Download results

```bash
cp /ptmp/paulkull/atoms_wor_baseline/partials/baseline.npz \
   /u/paulkull/pcla/data/WOR/baseline_data/baseline.npz
cp /ptmp/paulkull/atoms_wor_baseline/partials/mdx_features.npz \
   /u/paulkull/pcla/data/WOR/baseline_data/mdx_features.npz

cd /u/paulkull/pcla
git add -f data/WOR/baseline_data/baseline.npz
git add -f data/WOR/baseline_data/mdx_features.npz
git commit -m "add WOR baseline.npz and mdx_features.npz from HPC"
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
    /u/paulkull/pcla/pcla_agents/wor_pretrained/leaderboard_weights
```

Chains three SLURM jobs automatically:
1. **prep** — applies perturbations to both cameras → `test_labeled.npz`
2. **array** — 20-frame chunks; computes ATOMs profiles + 28-dim PEOC action logits
3. **gather** — concatenates → `test_profiles.npy` + `test_logits.npy`

Monitor: `squeue -u paulkull`

#### 3. Download results

```bash
cp /ptmp/paulkull/atoms_wor_test/test_profiles.npy \
   /u/paulkull/pcla/data/WOR/test_data/attention/test_profiles.npy
cp /ptmp/paulkull/atoms_wor_test/test_logits.npy \
   /u/paulkull/pcla/data/WOR/test_data/attention/test_logits.npy

cd /u/paulkull/pcla
git add -f data/WOR/test_data/attention/test_profiles.npy
git add -f data/WOR/test_data/attention/test_logits.npy
git commit -m "add WOR test_profiles.npy and test_logits.npy from HPC"
git push
```

Then locally: `git pull`, set `RECOMPUTE_TEST_ATOMS = False` in `atoms_config.py`.

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
    pgd
```

Chains three SLURM jobs automatically:
1. **prep** — concatenates `run_pgd_live_pert_*.npz` files, preserving both cameras → `live_pert_concat.npz`
2. **array** — 10 parallel ATOMs tasks (20 frames each), also computing 28-dim PEOC action logits
3. **gather** — concatenates results → `live_pert_profiles.npy` + `live_pert_action_logits.npy`

Monitor: `squeue -u paulkull`

#### 3. Download results

```bash
PERT=pgd
ATT=/u/paulkull/pcla/data/WOR/test_data/attention/live_pert/$PERT
mkdir -p $ATT

cp /ptmp/paulkull/atoms_wor_live_pert/live_pert_profiles.npy       $ATT/live_pert_profiles.npy
cp /ptmp/paulkull/atoms_wor_live_pert/live_pert_action_logits.npy  $ATT/live_pert_action_logits.npy

cd /u/paulkull/pcla
git add -f data/WOR/test_data/attention/live_pert/$PERT/live_pert_profiles.npy
git add -f data/WOR/test_data/attention/live_pert/$PERT/live_pert_action_logits.npy
git commit -m "add WOR live_pert_profiles for $PERT from HPC"
git push
```

Then locally: `git pull`, and set `RECOMPUTE_TEST_ATOMS = False` in `atoms_config.py`.
