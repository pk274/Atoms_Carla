# Topic 12 — Visualization: the figure factory (`visualization_carla.py`), the thesis-wide style system (`viz_config.py`), and the figure→thesis-section map

All claims verified against code on 2026-06-14. Line numbers refer to the current working tree.
Primary sources read in full: `ATOMs_Analysis/utils/visualization_carla.py` (1473 lines), `ATOMs_Analysis/utils/viz_config.py` (319 lines). Callers verified against `run_analysis.py` (STEP 5/6/7/14 figure calls, `:85-98,471-664,1709-1885`), `run_online_analysis.py` (`:77-78,309-310,447-518,660-687`), `ATOMs_Analysis/detection/baseline_dataset.py` (`:57,526-548`), `pcla_agents/wor/image_agent.py:19`, `ATOMs_Analysis/saliency/atoms_carla.py:47`. Cross-checked against `documentation/04_atoms.md`, `08_offline_analysis.md` (STEP table, `make_output_dirs`), `09_online_analysis.md`, `11_validation_and_testing.md`, `CLAUDE.md` (Data Layout, viz module note).

This is the **last topic file**; a whole-folder consistency pass (cross-references, terminology, contradictions) follows after it.

---

## 1. Purpose & scope

This document covers the **output layer** of the project — the single module that turns every numerical artefact produced by Topics 3–9 into a saved figure, plus the central style module that makes those figures visually coherent across the thesis. Two files carry the entire visualization responsibility:

1. **`visualization_carla.py`** — the *figure factory*. ~30 functions that take arrays (relevance maps, ATOMs profiles, GMM cluster labels, detector score arrays, ROC results, BIC/AIC sweeps, displacement statistics) and return `matplotlib.Figure` objects (or, for the three save-internally functions, write a PNG directly). It owns the relevance/segmentation overlays (Topic 3/4 output), the PCA/t-SNE embeddings (baseline/cluster/OOD), the attention bar charts (Topic 4 profiles), the model-selection and k-NN-sensitivity plots (Topic 7/8), the ROC + score-distribution plots (Topic 7 detectors), the (dead) trajectory/displacement family (the disabled STEP 10 of Topic 8), and the online distance-over-time plot (Topic 9).
2. **`viz_config.py`** — the *style system*. A flat module of constants (colours, figure sizes, marker/alpha/line styles, font sizes, save DPI) plus `apply_default_style()` (one rcParams push) and two semantic colour-lookup helpers (`get_perturbation_color`, the `DISTANCE_TYPE_COLORS`/`_YLABELS` dicts). It is **deliberately maintained identical to the Atari sister project** (module docstring `:6-10`) so any figure showing the same concept (baseline cloud, a `gaussian_noise` perturbation, a Mahalanobis distance) uses the same colour and size in both domains — a thesis-presentation requirement, not a code requirement.

This topic does **not** re-derive what each plotted quantity *means* (that is Topics 3–9); it documents *what each plot shows, what it consumes, which pipeline step calls it, where it is written, and which agent it applies to*, plus the complete style-constant inventory (hardcoded vs configurable) and the figure→thesis-section mapping the student needs. The single most load-bearing structural facts here, established below, are: (a) the `CARLA_CLASSES`/`TFV6_CLASSES` class tables are **duplicated** in this module (finding 4.9 confirmed, `:57-104`); (b) the entire trajectory/displacement plotting family is **dead** because its only callers (STEP 10) are commented out (finding 8.3); and (c) a sizeable fraction of `viz_config.py`'s constants exist only for the Atari project and are **unused in CARLA** (`FIGSIZE_BAR`, the `TRAJ_*` family).

---

## 2. Key design decisions

### 2.1 Centralised styling via `viz_config.apply_default_style()`, applied on import

`visualization_carla.py` calls `vc.apply_default_style()` **at import time** (`:52`), not inside any function. The rcParams push (`viz_config.py:262-276`) sets the thesis-wide fonts (title 13/bold, axis-label 11, tick/legend 9), the save DPI (200), `savefig.bbox="tight"`, grid off by default, and top/right spines off. Consequently *any* figure built after this module is imported — including figures built by code that never imports `viz_config` directly — inherits the style. The design intent is that a script need only `import` something from `visualization_carla` and every subsequent `plt.subplots()` is already thesis-styled. Per-call style overrides (figure size, legend font, marker size) are then read explicitly from `vc.*` constants inside each function, so the *global* rcParams set the typography and the *per-function* constants set the geometry/colour. This two-layer split (rcParams for type, named constants for layout) is the core styling decision.

### 2.2 Figure-returns-`Figure` + `save_figure(fig, path)` separation of concerns

The module docstring states the contract explicitly (`:28-29`): "All functions return matplotlib Figure objects so they can be saved or shown by the caller — no `plt.show()` calls inside." The *builders* (`plot_pca_baseline`, `plot_roc`, `plot_attention_bar`, …) construct and return a `Figure`; the *writer* `save_figure(fig, path, dpi=None)` (`:994-1002`) creates parent dirs, saves at `vc.SAVE_DPI`/`vc.SAVE_BBOX_INCHES`, closes the figure, and prints a confirmation. This keeps the builders pure and testable, lets the caller decide the output path (so the same builder serves baseline and per-perturbation variants), and centralises the DPI/bbox policy in one place. **Three functions break this contract** and save internally (Topic 12 finding 12.4): `visualize_relevance`, `visualize_comparative_relevance`, `visualize_segmentation` take a `save_path` and call `plt.savefig`/`plt.close` themselves (they return `None`); and `plot_distance_over_time` (`:1359-1414`) builds *and* saves *and* names its own file. The split is therefore observed by the analysis-figure family but not by the per-frame overlay family or the online distance plot.

### 2.3 Semantic colour conventions: one colour per concept, four colour families

`viz_config.py` defines four parallel colour conventions, each keyed by a *concept* so the same concept is the same colour everywhere:

- **Semantic role colours** (`:28-40`): baseline cloud `#9aa0a6` (muted grey, deliberately recessive), clean-test `tab:blue`, perturbed/OOD `tab:red`, threshold/chance lines `black`. Used by every PCA/t-SNE/score-distribution plot.
- **Per-perturbation colours** (`PERTURBATION_COLORS`, `:53-75`): a 14-entry dict mapping each perturbation name to a fixed colour, with `get_perturbation_color(name, fallback_index)` (`:80-84`) falling back to a stable `tab10` index for unknown names. Crucially, the dict carries **both projects' names** (e.g. `salt_pepper_noise` and `salt_and_pepper` map to the same `tab:orange`) so cross-domain figures match. Of these, only `gaussian_noise`, `brightness_scale`, `camera_loss`, `pgd_attack`/`fgsm_attack` are CARLA-relevant (Topic 6); the rest are Atari-only or aspirational (finding 12.5).
- **Per-cluster colours** (`CLUSTER_CMAP="tab10"`, `CLUSTER_CMAP_LARGE="tab20"`, `:91-92`): the GMM-cluster scatter colormap, with `get_cluster_colors(k)` (`:451-462`) returning the *exact same* K colours so the attention bar charts can be coloured to match the cluster scatter (a deliberate "linked styling" decision — Step 7 passes `get_cluster_colors(N_COMPONENTS)` into `plot_attention_comparison`, `run_analysis.py:498,545-552`).
- **Per-detector / per-distance-type colours** (`DISTANCE_TYPE_COLORS`, `:115-128`; `DISTANCE_TYPE_YLABELS`, `:130-143`): one colour and one axis-label per distance metric, with GMM variants given darkened shades of their single-Gaussian twin (e.g. `mahalanobis="tab:red"`, `gmm_mahalanobis="#810000"`). `plot_roc` and `plot_distance_over_time` look these up by **substring-matching the detector display name** (`_distance_type_from_detector_name`, `:1416-1441`; `_get_plot_style`, `:1444-1472`), so a name like `"ATOMs-Mahalanobis (GMM K=5)"` resolves to `gmm_mahalanobis`. Names that match no key fall back to matplotlib's default `C0…C9` cycle (ROC) — so detector colours are consistent across all ROC views without per-call colour bookkeeping.

### 2.4 Three embedding choices: PCA, t-SNE, and whitened-PCA — each with a rationale

The module offers three dimensionality-reduction paths, used for different purposes:

- **PCA** (`fit_pca`, `:258-263`): linear, *parametric* (a fitted `PCA` object can `.transform` new points). This is why the OOD overlay (`plot_pca_ood`) can project test points into the **same** baseline PCA space — `run_analysis.py:640` fits `pca_obj` once on the baseline and reuses it for every OOD overlay so all scatter plots share one geometry. PCA axes carry an interpretable explained-variance-ratio that is printed on the axis labels.
- **t-SNE** (`fit_tsne`, `fit_tsne_joint`, `:266-293`): non-linear, *non-parametric* — new points cannot be projected onto an existing embedding (the docstring spells this out, `:272-273`). The workaround for OOD is `fit_tsne_joint(baseline, test)` (`:280-293`), which fits one t-SNE on the concatenation and slices the first `N_b` rows as baseline; `run_analysis.py:1860` computes this once and reuses it across all per-perturbation t-SNE plots to avoid repeated expensive fits.
- **Whitened PCA** (`fit_whitened_pca`/`apply_whitened_pca`, `:1034-1076`): PCA on the *supplied covariance matrix* with a `λ^{-1/2}` scaling, so Euclidean distance in the projected plot equals **Mahalanobis** distance in feature space — the geometrically correct view for Mahalanobis-based detectors. This is used only by the (dead) trajectory plot (`:1167-1177`); it is implemented and correct but currently unreachable (§2.5).

t-SNE uses a fixed `random_state=42` (`:276`), PCA likewise (`:261`) — so embeddings are deterministic run-to-run, but t-SNE's `perplexity`/`n_iter` are never set (sklearn defaults; finding 12.6).

### 2.5 The trajectory/displacement plotting family is present but dead

Five functions form a self-contained "perturbation trajectory in attention space" family: `compute_perturbation_displacement_stats` (`:1083-1121`), `format_displacement_stats_text` (`:1124-1134`), `plot_pca_perturbation_trajectories` (`:1141-1293`), `plot_displacement_coherence_bar` (`:1296-1328`), `plot_displacement_magnitude_boxplot` (`:1331-1356`). They are fully implemented (the trajectory plot even does whitened-PCA, individual + mean displacement arrows, and a "PC1 top drivers" subtitle). **Their only callers are the commented-out STEP 10 block of `run_analysis.py` (`:1057-1104`, every line `#`-prefixed)** — Topic 8 §2.4 confirmed the entire trajectory step is disabled. Therefore these five functions are dead code, no `trajectory_*.png`/`displacement_*.png` is produced by the current pipeline, and any such figures on disk are **stale** (finding 8.3). They are documented here for completeness and because the thesis text may still reference the *concept* (displacement coherence R, mean population shift) even though the figures are not regenerated.

### 2.6 Duplicated class tables (finding 4.9, confirmed here)

`CARLA_CLASSES` (29 entries, tags 0–28) and `TFV6_CLASSES` (10 grouped entries) are defined in this module at `:57-104` — and *independently* in `ATOMs_Analysis/saliency/atoms_carla.py:57-103`. The two copies are currently identical, but they are maintained separately: local pipelines import the class map from `atoms_carla`, HPC chunk scripts import from `visualization_carla`, so a future edit to one and not the other would silently change profile/legend semantics in one path. In *this* module the tables feed only `visualize_segmentation`'s legend (`:211,237`) and serve as the default `class_map` — the ATOMs profile dimensionality itself comes from `atoms_carla`'s copy. (Note the docstring at `:91-92` says LEAD "collapses the 32 raw CARLA classes into 10" while `CARLA_CLASSES` here lists 29 — the perennial 23/29/32 tag-count drift, cf. findings 1.12, 4.1.)

---

## 3. Implementation details

### 3.1 Figure inventory (grouped by family)

Every plotting/visualization entry point, what it shows, the pipeline step that calls it, the output filename/subdir, the agent it applies to, and its file:line. Filenames are relative to `OUT_DIR = RESULTS_DIR/atoms_analysis_mode_<mode>/` (offline) or the online `OUT_DIR = RESULTS_DIR/atoms_analysis_live_mode_<mode>/` (`run_online_analysis.py:89`). "Step" uses the in-code STEP banners of Topic 8 §3.1.

**Family A — relevance & segmentation overlays (per-frame, save-internally)**

| Function | What it shows | Called by | Output | Agent | Line |
|---|---|---|---|---|---|
| `visualize_relevance` | One-sided (positive) relevance heatmap + RGB overlay; brake frames use an alt colormap | online STEP 8 (`run_online_analysis.py:447,517`); `baseline_dataset.py:548` (offline, gated `PLOT_SEG_AND_REL`) | `<REL_DIR>/relevance_wide_<i>` (no `.png` ext — finding 12.7) | both | 107 |
| `visualize_comparative_relevance` | Signed *drive − brake* relevance with diverging seismic cmap, per-pixel alpha ∝ \|signal\| | online STEP 8 (`run_online_analysis.py:463,476`); imported in `baseline_dataset.py:57` | `…_comparative` | both (needs `PLOT_COMPARATIVE_REL`) | 149 |
| `visualize_segmentation` | Discrete-coloured semantic map (`tab20`) with a legend of only the classes present | `baseline_dataset.py:526,528` (offline, gated) | `<…>/segmentation_*` | both (class_map from `atoms.class_map`) | 192 |

**Family B — embeddings (PCA / t-SNE × baseline / clusters / OOD)**

| Function | What it shows | Called by (step) | Output (subdir) | Agent | Line |
|---|---|---|---|---|---|
| `plot_pca_baseline` | PCA scatter of baseline ATOMs profiles, coloured by run_id (or cmd) | STEP 7 (`run_analysis.py:618`) | `pca/pca_baseline_by_run.png` | both | 300 |
| `plot_tsne_baseline` | t-SNE counterpart | STEP 7 (`:648`) | `tsne/tsne_baseline_by_run.png` | both | 351 |
| `plot_pca_clusters` | PCA coloured by GMM cluster; optional centroids as stars | STEP 7 (`:630`) | `pca/pca_baseline_clusters.png` | both | 391 |
| `plot_tsne_clusters` | t-SNE counterpart | STEP 7 (`:658`) | `tsne/tsne_baseline_clusters.png` | both | 465 |
| `plot_pca_ood` | Baseline (grey) + clean test (blue) + perturbed (red), reusing the baseline PCA | STEP 14 (`:1832,1848`) | `pca/pca_ood_overlay.png`, `pca/pca_ood_<pert>.png` | both | 499 |
| `plot_tsne_ood` | t-SNE counterpart via joint embedding | STEP 14 (`:1862,1878`) | `tsne/tsne_ood_overlay.png`, `tsne/tsne_ood_<pert>.png` | both | 562 |
| `fit_pca` / `fit_tsne` / `fit_tsne_joint` | DR helpers (not plots) | STEP 6/7/14 (`:640,646,1860`) | — | both | 258 / 266 / 280 |
| `get_cluster_colors` | K colours matching the cluster colormap (for linked bar styling) | STEP 6 (`:498`) | — | both | 451 |

**Family C — attention bar charts (Topic 4 profiles)**

| Function | What it shows | Called by (step) | Output (subdir) | Agent | Line |
|---|---|---|---|---|---|
| `plot_attention_bar` | Horizontal bar of mean normalized attention per class (optional top-k, error bars) | STEP 7 (`run_analysis.py:520`) | `attention/baseline_attention_bar.png` | both | 611 |
| `plot_attention_comparison` | Grouped bars comparing profiles across conditions (clusters or commands); accepts linked cluster colours | STEP 7 (`:545` clusters, `:588` per-cmd) | `attention/attention_by_cluster.png`, `attention/attention_by_command.png` (cmd only if >1 present) | both | 658 |
| `plot_attention_bars_separate` | One identically-styled bar chart per condition (for collages), shared top-k, optional min–max whiskers | STEP 7 (`:557-568`) | `attention/<cluster_label>_attention_bar.png` | both | 710 |
| `plot_cluster_representative` | Wraps one [3,H,W] RGB frame (nearest to a cluster mean) in a labelled figure | STEP 7 (`:611`) | `attention/cluster_<k>_representative.png` | both | 785 |

**Family D — detector evaluation (ROC + score distributions)**

| Function | What it shows | Called by (step) | Output (subdir) | Agent | Line |
|---|---|---|---|---|---|
| `plot_roc` | ROC curves for one/many detectors, colour-keyed by detector name, Youden point marked, sorted by AUC | STEP 14 (`run_analysis.py:1709,1720,1732,1740,1782,1789,1796`) | `roc/roc_all_detectors.png`, `roc_static_detectors.png`, `roc_gmm_vs_static.png`, `roc_top5.png`, `roc_<pert>_*.png` | both | 819 |
| `plot_mahal_distribution` | Overlapping in-vs-out score histograms (density), optional threshold line | STEP 14 (`:1760,1769`) | `scores/score_dist_<detector>_single.png`, `…_gmm.png` | both | 860 |

**Family E — model selection & hyperparameter sensitivity (Topic 7/8)**

| Function | What it shows | Called by (step) | Output (subdir) | Agent | Line |
|---|---|---|---|---|---|
| `plot_bic_aic` | BIC & AIC vs K, best-K marked for each | STEP 5 (`run_analysis.py:471`); online (`run_online_analysis.py:309`) | `clustering/gmm_model_selection.png` (offline); `gmm_model_selection.png` at OUT_DIR root (online) | both | 895 |
| `plot_knn_sensitivity` | AUC vs k bar chart, best-k highlighted in the k-NN colour, AUC annotated; shows **val** AUC when available | STEP 14 (`:1814,1817`) | `roc/knn_k_sensitivity.png`, `roc/knn_gmm_k_sensitivity.png` | both | 928 |

**Family F — trajectory / displacement (DEAD — disabled STEP 10, finding 8.3)**

| Function | What it shows | Called by | Output | Agent | Line |
|---|---|---|---|---|---|
| `compute_perturbation_displacement_stats` | Per-perturbation displacement vectors, magnitudes, within-type coherence R (not a plot) | STEP 10 (commented, `run_analysis.py:1057`) | — | both | 1083 |
| `format_displacement_stats_text` | Human-readable stats block | STEP 10 (commented, `:1059`) | `trajectory_analysis/displacement_stats.txt` (never written) | both | 1124 |
| `plot_pca_perturbation_trajectories` | Clean→perturbed arrows + mean shift in (whitened) PCA space | STEP 10 (commented, `:1075,1087`) | `trajectory_analysis/trajectory_pca.png`, `trajectory_whitened_pca.png` | both | 1141 |
| `plot_displacement_coherence_bar` | Bar of within-type coherence R per perturbation | STEP 10 (commented, `:1099`) | `trajectory_analysis/displacement_coherence.png` | both | 1296 |
| `plot_displacement_magnitude_boxplot` | Box plot of L2 displacement magnitudes per perturbation | STEP 10 (commented, `:1103`) | `trajectory_analysis/displacement_magnitude.png` | both | 1331 |

**Family G — online distance-over-time (Topic 9)**

| Function | What it shows | Called by | Output | Agent | Line |
|---|---|---|---|---|---|
| `plot_distance_over_time` | Per-frame distance vs frame index (`"o-"`, **no smoothing**), optional clean overlay + injection line; builds+saves+names internally | online (`run_online_analysis.py:677-685`) | `<OUT_DIR>/<distance_type>_<pert>_<variant>.png` | both | 1359 |

**Helpers (non-figure)**: `save_figure` (`:994`), `make_output_dirs` (`:1005`), `fit_whitened_pca`/`apply_whitened_pca` (`:1034/1069`), `_distance_type_from_detector_name` (`:1416`), `_get_plot_style` (`:1444`).

### 3.2 `viz_config.py` style system in detail

**`apply_default_style()` (`:254-276`)** pushes nine rcParams: `figure.dpi=100` (screen preview), `savefig.dpi=SAVE_DPI` (200), `savefig.bbox="tight"`, `axes.titlesize=13`/`axes.titleweight="bold"`, `axes.labelsize=11`, `xtick.labelsize=ytick.labelsize=9`, `legend.fontsize=9`, `legend.framealpha=0.85`, `axes.grid=False`, and top/right spines off. It sets typography + save policy globally; geometry and colour stay in the per-function constants.

**Colour palettes** (counts): semantic-role colours = 5 (`:28-40`); `PERTURBATION_COLORS` = 14 entries spanning both projects (`:53-75`); cluster colormaps = `tab10`/`tab20` (10/20 discrete colours, `:91-92`); `DISTANCE_TYPE_COLORS` = 12 keys with matching 12-key `DISTANCE_TYPE_YLABELS` (`:115-143`); saliency colormaps = `hot` (positive), `gist_earth` (brake-alt), `seismic` (signed diverging) (`:104-106`); centroid styling = star marker, size 250, black edge (`:94-97`).

**Figure-size constants** (`:150-173`): `FIGSIZE_PCA=(8,6)` (also reused for trajectory, `FIGSIZE_TRAJECTORY=(8,6)`), `FIGSIZE_ROC=(5,5)` square, `FIGSIZE_HISTOGRAM=(6,4)`, `FIGSIZE_BIC_AIC=(5,3.5)`, `FIGSIZE_BAR=(7,4)` (**defined but never used in CARLA** — bar charts use the two scaling helpers instead; finding 12.5), `FIGSIZE_ATTENTION_OVERLAY=(15,4)` (relevance triptych — though the actual triptych is built as a 1×2 subplot, §3.3), `FIGSIZE_DISTANCE_OVER_TIME=(7,5)`. Two helper functions size bar charts dynamically: `figsize_bar_scaled(n_items, per_item, min_w, height)` (width grows with bar count) and `figsize_attention_bar(n_classes, per_class=0.35, width=7, min_h=3)` (height grows with class count).

**Marker/alpha/line constants** (`:180-216`): baseline marker size 14 / alpha 0.40; clean & perturbed marker size 18 / alpha 0.65; the entire `TRAJ_*` block (point size 30, line width 1.5, start/end markers, endpoint size 80) — the `TRAJ_POINT_SIZE`/`TRAJ_LINE_WIDTH`/`TRAJ_ARROW_LINEWIDTH`/`TRAJ_START_MARKER`/`TRAJ_END_MARKER`/`TRAJ_ENDPOINT_SIZE` constants are **Atari-only and never read in CARLA** (only `TRAJ_ENDPOINT_EDGECOLOR`/`_LINEWIDTH` are used, by the dead trajectory plot; finding 12.5); the displacement-arrow family (`ARROW_ALPHA_INDIVIDUAL=0.25`, `ARROW_LINEWIDTH_MEAN=2.5`, mean/clean/perturbed marker sizes) is read only by the dead `plot_pca_perturbation_trajectories`.

**Fonts / saving / grid** (`:223-247`): title 13 bold, subtitle 10, axis-label 11, tick/legend 9; `SAVE_DPI=200`, `SAVE_BBOX_INCHES="tight"`, `SAVE_FORMAT_DEFAULT="png"` (the format constant is never actually consulted — callers hardcode `.png` suffixes); `GRID_LINESTYLE="--"`, `GRID_ALPHA=0.3`, `LEGEND_FRAMEALPHA=0.85`. An `__all__` (`:279-318`) re-exports the public surface.

### 3.3 How `visualization_carla.py` consumes the style system

Every builder reads `vc.*` rather than hardcoding. Examples: `plot_pca_baseline` uses `vc.FIGSIZE_PCA`, `vc.BASELINE_MARKER_SIZE/_ALPHA`, `vc.BASELINE_COLOR`, `vc.CLUSTER_CMAP` (`:326-341`); `plot_roc` uses `vc.FIGSIZE_ROC`, `vc.CHANCE_LINE_COLOR`, and resolves each curve colour via `DISTANCE_TYPE_COLORS[_distance_type_from_detector_name(name)]` (`:831-850`); `plot_attention_bar` sizes itself with `vc.figsize_attention_bar(len(idx))` and defaults its colour to `vc.CLEAN_TEST_COLOR` (`:634-648`); `visualize_relevance` reads `vc.SALIENCY_OVERLAY_ALPHA`, `vc.SALIENCY_CMAP_POSITIVE`/`_ALT`, and `vc.SAVE_DPI` (`:113-143`). The relevance "triptych" docstring (Input | … | Overlay) and `FIGSIZE_ATTENTION_OVERLAY=(15,4)` describe a 3-panel layout, but the code actually builds a 1×2 subplot (Relevance | Overlay) at that size (`:134-135`) — the input panel is not drawn (finding 12.8).

### 3.4 `save_figure` and `make_output_dirs`

`save_figure(fig, path, dpi=None)` (`:994-1002`): `dpi` defaults to `vc.SAVE_DPI`; creates `path.parent` (so callers need not pre-create), saves with `bbox_inches=vc.SAVE_BBOX_INCHES`, closes the figure, prints `[visualization] Saved → …`. `make_output_dirs(base_dir)` (`:1005-1027`) creates exactly six subdirs — `pca/`, `tsne/`, `roc/`, `attention/`, `scores/`, `clustering/` — and returns a `{name: Path}` dict so callers write `save_figure(fig, dirs["pca"]/"x.png")`. It does **not** create a `trajectory_analysis/` subdir (that was only ever created by the now-dead STEP 10), confirming finding 8.6. JSON results and saved detector models are written at the `OUT_DIR` root, not in any of these six subdirs (Topic 8 §3.14).

### 3.5 Colour-name resolution (`_distance_type_from_detector_name`, `_get_plot_style`)

Both helpers do **most-specific-first substring matching**: they test `"gmm" in name` and the metric keyword to pick e.g. `gmm_mahalanobis` before `mahalanobis` (`:1424-1441`, `:1453-1472`). `_distance_type_from_detector_name` returns `None` for non-distance detectors (Action entropy, MDX) so `plot_roc` falls back to the `C0…C9` cycle; `_get_plot_style` always returns a `(color, ylabel)` pair (defaulting to euclidean) and raises `KeyError` only if a key is somehow absent from `DISTANCE_TYPE_COLORS`/`_YLABELS`. The two functions duplicate the same matching ladder (one returns a key-or-None, the other a style tuple) — a small redundancy that could drift (finding 12.9).

### 3.6 Online vs offline figure routing

Offline (`run_analysis.py`) writes into the `make_output_dirs` six-subdir tree under `atoms_analysis_mode_<mode>/`. Online (`run_online_analysis.py`) writes into a flat `atoms_analysis_live_mode_<mode>/` directory: `plot_bic_aic` → `gmm_model_selection.png` at the root (`:309-310`); `visualize_relevance`/`visualize_comparative_relevance` per-frame overlays → `TEST_DATA_DIR/relevance_live_pert/<pert>/<variant>/relevance_wide_<i>` (`:423,441-465`); `plot_distance_over_time` → `<OUT_DIR>/<distance_type>_<pert>_<variant>.png` (`:677-685`). A `live_pert_dir = RESULTS_DIR/live_perturbation/<pert>` is created every loop iteration (`:660-661`) but **never written to** — the figures go to `OUT_DIR` — so it is dead directory creation (finding 9.10, restated here as it is a visualization-routing bug).

---

## 4. Parameters & magic constants

All in `viz_config.py` unless noted. "Configurable" = exposed in `atoms_config.py`; "code" = hardcoded literal in a viz file; "hardcoded-caller" = literal passed at a call site.

| Constant | Value | Where | Configurable? | Meaning |
|---|---|---|---|---|
| `SAVE_DPI` | 200 | `viz_config.py:236`; used by every save | code | output figure DPI |
| `figure.dpi` (preview) | 100 | `viz_config.py:263` | code | screen-preview DPI (rcParam) |
| `SAVE_BBOX_INCHES` | "tight" | `:237` | code | savefig bbox |
| `SAVE_FORMAT_DEFAULT` | "png" | `:238` | code | **declared, never consulted** (callers hardcode `.png`) |
| `FONTSIZE_TITLE` / weight | 13 / "bold" | `:223,229` | code | title type (rcParam) |
| `FONTSIZE_SUBTITLE` | 10 | `:224` | code | trajectory "PC1 drivers" subtitle |
| `FONTSIZE_AXIS_LABEL` | 11 | `:225` | code | axis labels (rcParam) |
| `FONTSIZE_TICK` / `_LEGEND` | 9 / 9 | `:226-227` | code | ticks, legends, bar annotations |
| `FIGSIZE_PCA` / `_TRAJECTORY` | (8,6) / (8,6) | `:150-151` | code | PCA, t-SNE, OOD, trajectory scatters |
| `FIGSIZE_ROC` | (5,5) | `:152` | code | ROC (square) |
| `FIGSIZE_HISTOGRAM` | (6,4) | `:153` | code | score-distribution histograms |
| `FIGSIZE_BIC_AIC` | (5,3.5) | `:154` | code | model-selection plot |
| `FIGSIZE_BAR` | (7,4) | `:155` | code | **defined, unused in CARLA** (finding 12.5) |
| `FIGSIZE_ATTENTION_OVERLAY` | (15,4) | `:156` | code | relevance overlay (built as 1×2, finding 12.8) |
| `FIGSIZE_DISTANCE_OVER_TIME` | (7,5) | `:157` | code | online distance-vs-frame |
| `figsize_attention_bar` per_class | 0.35 (h grows) | `:168-173` | code | attention-bar height scaling |
| `figsize_bar_scaled` per_item | caller-set (0.8/0.9/1.1/1.2) | `:160-165`; callers `vis.:691,961,1313,1344` | code | bar-width scaling |
| `BASELINE_MARKER_SIZE`/`_ALPHA` | 14 / 0.40 | `:181-182` | code | baseline cloud |
| `CLEAN`/`PERTURBED_MARKER_SIZE`/`_ALPHA` | 18 / 0.65 | `:185-190` | code | test points |
| cluster scatter alpha bump | +0.2 | `vis.:423,489` | code | clusters drawn more saturated than baseline |
| `SALIENCY_OVERLAY_ALPHA` | 0.5 | `:108` | code | relevance overlay blend |
| per-pixel comparative alpha exp | 0.99 | `vis.:179` | code | \|signal\|^0.99 opacity (signed map) |
| `CLUSTER_CMAP` / `_LARGE` | tab10 / tab20 | `:91-92` | code | ≤10 vs >10 clusters (finding 12.2 risk) |
| `CENTROID_SIZE` / marker | 250 / "*" | `:94-95` | code | GMM centroid stars |
| `PERTURBATION_COLORS` | 14 entries | `:53-75` | code | per-perturbation colours (both projects) |
| `DISTANCE_TYPE_COLORS`/`_YLABELS` | 12 keys each | `:115-143` | code | per-detector colour + axis label |
| PCA `random_state` | 42 | `vis.:261` | code | deterministic PCA |
| t-SNE `random_state` | 42 | `vis.:276` | code | deterministic t-SNE seed |
| t-SNE perplexity / n_iter | sklearn default | (never set) | — | **not configured** (finding 12.6) |
| PCA/t-SNE `n_components` | 2 | `vis.:258,266` etc. | code | embedding dim |
| trajectory subsample | 60 | `vis.:1146` | code | individual arrows per perturbation (dead) |
| trajectory arrow RNG seed | 0 | `vis.:1205` | code | subsample RNG (dead) |
| `plot_mahal_distribution` bins | 50 | `vis.:866` | code | histogram bins |
| whitened-PCA eigen floor | 1e-12 | `vis.:1057` | code | eigenvalue clamp |
| `PLOT_SEG_AND_REL` | True | `atoms_config.py:48`; `baseline_dataset.py` | config | gate offline seg+relevance overlays |
| `PLOT_COMPARATIVE_REL` | True | `atoms_config.py:49`; online `:450` | config | gate drive−brake comparative maps |
| `PLOT_INTERVAL` | 20 | `atoms_config.py:50` | config | frame stride for plotted overlays |
| `MODE_ANALYSIS` | 2 | `atoms_config.py:23` | config | suffixes `OUT_DIR` name |

---

## 5. Known limitations & open issues

- **Duplicated `CARLA_CLASSES`/`TFV6_CLASSES` (finding 4.9, confirmed `:57-104`).** Identical to `atoms_carla.py:57-103` but maintained independently; local code imports the class map from `atoms_carla`, HPC scripts from here — silent divergence risk. Only the legend in `visualize_segmentation` consumes this module's copy. The docstring's "32 raw CARLA classes" vs the 29-entry `CARLA_CLASSES` continues the tag-count drift (1.12, 4.1).
- **Trajectory/displacement family is dead; on-disk figures stale (finding 8.3).** Five fully-implemented functions (`:1083-1356`) whose only callers are the commented STEP 10 block. No `trajectory_*`/`displacement_*` figure is regenerated; `make_output_dirs` no longer creates the `trajectory_analysis/` subdir (finding 8.6). `fit_whitened_pca`/`apply_whitened_pca` are correct but reachable only through this dead family.
- **No score smoothing in `plot_distance_over_time` (finding 9.5).** Raw per-frame distances are plotted with `"o-"`; no rolling-mean/window/EWMA option, so the online traces are jittery and the injection change-point is hard to read. (Cross-ref Topic 9 §; the online experiment is qualitative-only, finding 9.6.)
- **Unused `live_pert_dir` (finding 9.10).** Created every online loop iteration (`run_online_analysis.py:660-661`) but never written to; explains the empty `results/live_perturbation/<pert>/` dirs on disk.
- **`save_path` written without a `.png` extension (Topic 12 finding 12.7).** The online overlay calls pass `REL_DIR/f"relevance_wide_{i}"` (no suffix) to `visualize_relevance` (`run_online_analysis.py:441-447,512-517`); `plt.savefig` then infers PNG from rcParams but the files land extension-less, so they do not open by double-click and are easy to mistake for non-images.
- **`save_figure`/builder contract broken by four functions (finding 12.4).** `visualize_relevance`, `visualize_comparative_relevance`, `visualize_segmentation`, and `plot_distance_over_time` save (and the last also names) internally and return `None`, unlike the documented "return a Figure" contract (`:28-29`). Mixed conventions make the module harder to reason about and to unit-test.
- **Unused Atari-only style constants (finding 12.5).** `FIGSIZE_BAR`, `TRAJ_POINT_SIZE`, `TRAJ_LINE_WIDTH`, `TRAJ_LINE_ALPHA`, `TRAJ_ARROW_LINEWIDTH`, `TRAJ_START_MARKER`, `TRAJ_END_MARKER`, `TRAJ_ENDPOINT_SIZE`, and ~8 `PERTURBATION_COLORS` keys (`blur`, `salt_*`, `*_brightness`, `random_occlusion`, `phantom_obstacle`) are never read by CARLA code — they exist solely to keep the file byte-identical to the Atari sister project. Harmless but a reader cannot tell live constants from cross-project ballast.
- **t-SNE perplexity / n_iter never set (finding 12.6).** Only `random_state=42` is fixed; `perplexity` and `n_iter` fall to sklearn defaults, which depend on N and the installed sklearn version — so t-SNE figures are deterministic for a fixed env but not reproducible across sklearn versions, and perplexity is not tuned to the baseline cloud size.
- **Cluster colormap can run out of colours for large K (finding 12.2).** `CLUSTER_CMAP_LARGE="tab20"` caps at 20 distinct colours; with K>20 (allowed by `--gmm-k`) clusters wrap and become indistinguishable in the scatter and in `get_cluster_colors`. No guard. Current sweeps stay ≤12 (Topic 8 §3.15), so latent.
- **Relevance "triptych" is actually a 1×2 panel (finding 12.8).** `visualize_relevance`'s docstring and `FIGSIZE_ATTENTION_OVERLAY=(15,4)` imply Input | Relevance | Overlay, but only Relevance | Overlay are drawn (`:134-140`); the input RGB is folded into the overlay, not shown separately. `visualize_comparative_relevance`'s docstring claims "Input | Drive − Brake | Overlay" but likewise builds 1×2 (`:171-182`).
- **Duplicated colour-resolution ladders (finding 12.9).** `_distance_type_from_detector_name` (`:1416`) and `_get_plot_style` (`:1444`) hardcode the same most-specific-first metric-matching ladder twice; an edit to the colour keys must be mirrored in both.
- **NameError risk in the offline overlay branch (finding 4.8, restated).** In `baseline_dataset.py:543-548`, `rgb_wide` is assigned only inside the `PLOT_COMPARATIVE_REL` branch but used unconditionally by `visualize_relevance` — `NameError` when `PLOT_SEG_AND_REL=True` and `PLOT_COMPARATIVE_REL=False`. The online loop (`run_online_analysis.py:442,513`) hoists `rgb_wide` correctly, so the bug is offline-only. Not re-numbered here.

---

## 6. Cross-references

- **01_architecture_overview.md** — `atoms_config.py` owns the only configurable viz flags (`PLOT_SEG_AND_REL`, `PLOT_COMPARATIVE_REL`, `PLOT_INTERVAL`, `MODE_ANALYSIS` which suffixes `OUT_DIR`); everything else (DPI, palettes, figure sizes, seeds) is hardcoded in `viz_config.py` — the visualization layer is largely decoupled from the central config. `EXPERIMENT_VARIANT` re-roots `RESULTS_DIR` and thus every figure's output path (finding 5.7).
- **02_agents.md** — figures are agent-agnostic in code but agent-specific in content: TFV6 uses `TFV6_CLASSES` (10) and 8-bin speed logits (PEOC colour), WoR uses `CARLA_CLASSES` (29) and 28-dim action logits (Action-entropy); `plot_cluster_representative` shows the wide-camera frame; `visualize_segmentation` legends from `atoms.class_map` (per-agent).
- **03_lrp.md** — `visualize_relevance` and `visualize_comparative_relevance` render the LRP output (positive heatmap; signed drive−brake map with the seismic diverging cmap matching the signed-relevance reality of finding 3.5/4.5). The "Qualitative LRP examples" figures of the thesis come from this family.
- **04_atoms.md** — the attention bar charts (Family C) and every embedding (Family B) plot the ATOMs profile (10-dim TFV6 / 29-dim WoR); `CARLA_CLASSES`/`TFV6_CLASSES` here duplicate `atoms_carla.py`'s (finding 4.9). Profiles fed to `plot_pca_*`/`plot_attention_*` are the per-frame hierarchical vectors of Topic 4.
- **07_distances_and_detectors.md** — `plot_roc`, `plot_mahal_distribution`, `plot_knn_sensitivity`, `plot_bic_aic` visualise the detectors, their score distributions, the k-NN k-sweep, and the GMM model selection; `DISTANCE_TYPE_COLORS`/`_YLABELS` provide one colour+label per detector. Wasserstein has colour/label keys (`:121,136`) but the detector is disabled (Topic 7 §5), so those keys are never exercised.
- **08_offline_analysis.md** — STEP 7 (baseline viz) and STEP 14 (detection figures) are the two call sites for almost all builders; `make_output_dirs`'s six-subdir tree and the mode-suffixed `OUT_DIR` (finding 8.6); the dead STEP 10 (finding 8.3) is the sole would-be caller of Family F.
- **09_online_analysis.md** — `plot_distance_over_time` is the only figure the online pipeline produces (Family G); no smoothing (9.5), unused `live_pert_dir` (9.10), qualitative-only / threshold-never-drawn (9.6), reduced detector roster (9.8).
- **10_hpc_pipeline.md** — HPC chunk scripts import the class tables from *this* module (the other half of finding 4.9); the QC figure `visualize_perturb.py` (finding 10.6) is a standalone HPC script, not part of this module.
- **11_validation_and_testing.md** — the validation harnesses emit `*_report.txt` + `*_per_frame.npy`, not figures; the only figure-adjacent QC is the HPC `visualize_perturb.py`. No test covers any function in `visualization_carla.py`.
- **99_bugs_and_findings.md** §"From Topic 12" — findings 12.1–12.9; cross-references 4.8, 4.9, 8.3, 8.6, 9.5, 9.6, 9.10, 10.6, and Topic 7 §5 (Wasserstein).

---

## 7. Figure → thesis-section mapping

The student-facing payoff: which thesis chapter/section each figure family supports, the figures it contributes, and the caveat to state.

| Figure family | Figures | Thesis section it supports | Caveat to note in the text |
|---|---|---|---|
| **A — relevance & segmentation overlays** | `relevance_wide_*`, `*_comparative`, `segmentation_*` | "Qualitative LRP / attribution examples" (Method illustration) | TFV6 maps are signed (seismic cmap); overlay is 1×2 not 3-panel (12.8); offline frames gated by `PLOT_SEG_AND_REL`/`PLOT_INTERVAL` |
| **B — PCA/t-SNE baseline & clusters** | `pca_baseline_by_run`, `pca_baseline_clusters`, `tsne_*` | "Baseline attention structure / clustering" (Results, exploratory) | PCA is reused for OOD overlays; t-SNE perplexity un-tuned (12.6) |
| **B — PCA/t-SNE OOD overlays** | `pca_ood_overlay`, `pca_ood_<pert>`, `tsne_ood_*` | "Detection results: separability of perturbed vs clean" (Results, qualitative) | clean test may overlap baseline (same distribution); per-pert overlays share one embedding |
| **C — attention bar charts** | `baseline_attention_bar`, `attention_by_cluster`, `attention_by_command`, per-cluster bars, `cluster_*_representative` | "What the agent attends to / per-cluster attention profiles" (Method + Results) | hierarchical-only ATOMs (finding 1.4/4.1); bars colour-linked to cluster scatter |
| **D — ROC + score distributions** | `roc_all_detectors`, `roc_gmm_vs_static`, `roc_top5`, `roc_<pert>_*`, `score_dist_*` | "Quantitative detection results / per-perturbation breakdown" (Results, main) | Youden point marked on ROC; per-pert ROC is clean-vs-one-perturbation |
| **E — model selection & k-NN sensitivity** | `gmm_model_selection` (BIC/AIC), `knn_k_sensitivity`, `knn_gmm_k_sensitivity` | "Hyperparameter selection (K, k)" (Method / Appendix) | K usually overridden by config (finding 7.1); k-NN sensitivity shows **val** AUC when present |
| **F — trajectory / displacement** | `trajectory_pca`, `trajectory_whitened_pca`, `displacement_coherence`, `displacement_magnitude` | "Attention displacement under perturbation" (would-be Results) | **DEAD — not regenerated (8.3); any on-disk figure is stale.** Use only if STEP 10 is re-enabled |
| **G — online distance-over-time** | `<distance_type>_<pert>_<variant>.png` | "Online detection case study" (Results, illustrative) | qualitative only, no AUC/threshold line (9.6); no smoothing (9.5) |
