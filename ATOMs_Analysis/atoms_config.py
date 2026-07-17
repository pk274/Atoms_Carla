class ExperimentConfig:
    """
    Central configuration for ATOMs experiments.
    """

    from pathlib import Path

    # Active agent — controls which data subfolder is used.
    # Accepted values: "WOR" (World on Rails) | "LBC" (Learning by Cheating) | "TFV6" (TransFuser v6)
    AGENT = "TFV6"

    # Number of cameras in the concatenated wide RGB image.
    # 6 → lead360 dataset (new, 6-camera; each camera 384×384 px → total 384×2304)
    # 3 → legacy 3-camera dataset (384×1152); kept for archival comparison only.
    # Changing this does NOT re-derive any paths — it is used for validation and
    # visualization only.  The analysis code auto-detects width from the npz data.
    N_CAMERAS = 6

    TOWN = "Town05"
    WEATHER = "sunny"
    SPEED_MODE = False
    HIGH_SPEED_MODE = False

    BASELINE_RECORDING_MODE = False
    TESTSET_RECORDING_MODE = False
    LIVE_PERTURBATION_RECORDING_MODE = True

    NUM_GMM_CLUSTERS = 8        # None for automatic BIC selection; overridden by --gmm-k CLI arg

    MODE_ANALYSIS = 2
    FC_RELEVANCE_FILTER = 0.9       # 0.9


    NOISE_INTENSITY = 25        # must match hpc/prep_test.py / prep_test_wor.py --noise-intensity default
    BRIGHTNESS_INTENSITY = 3

    PERTURBATION = "brightness_scale"
    INTENSITY = 3                 # 4 for brightness, 25 gn, 0.07 for po
    INJECTION_TIME = 10            # 10 for live perturbation
    AFFECT_BOTH_CAMS = True
    CAM_INDEX = None      # None for all cams; list for a subset (0-based: 0=front-left,1=front-center,2=front-right for 6-cam TFV6)
    MANUAL_SPAWNS = True

    RECOMPUTE_BASELINE = False
    RECOMPUTE_TEST_ATOMS = False
    REAPPLY_PERTURBATIONS = False
    RECOMPUTE_MDX_BASELINE      = False
    RECOMPUTE_MDX_V2_BASELINE   = False    # set False after first successful run
    RECOMPUTE_MDX_TEST_SCORES   = False    # set True to re-extract backbone features on test set

    ENABLE_MDX_V2 = False   # set True to fit/load and score with the MDX-v2 detector

    # MDX-v2 ablation flags — toggle independently to isolate which change helps
    MDX2_USE_FC_FEATURES      = True    # True: 256-d speed_query; False: 512-d backbone (like v1)
    MDX2_USE_QUANTILE_BINNING = True    # True: quantile bin edges; False: equal-width (like v1)

    PLOT_SEG_AND_REL = True
    PLOT_COMPARATIVE_REL = True
    PLOT_INTERVAL = 20           # 20
    PLOT_CLEAN_ATTENTION_OVERLAY = True  # overlay clean distance as faint line in distance-over-time plots


    IMAGE_SAMPLE_INTERVAL = 25   # 25
    TEST_SAMPLE_INTERVAL = 5     # 11
    MAX_BASELINE_SIZE = 100      # 100
    MAX_TEST_SIZE = 200
    MAX_LIVE_PERT_SIZE = 100
    _DATA_ROOT = Path("C:/Users/paulk/Desktop/Unistuff/Masterarbeit/Code/PCLA/data") / AGENT

    # Switch to "alternative" to use the same-distribution split (all towns,
    # random route-level split into *_data_alt directories).
    # "original" keeps the Town05-held-out split unchanged.
    EXPERIMENT_VARIANT = "alternative"   # "original" | "alternative"

    if EXPERIMENT_VARIANT == "alternative":
        BASELINE_DATA_DIR = _DATA_ROOT / "baseline_data_alt"
        TEST_DATA_DIR     = _DATA_ROOT / "test_data_alt"
        VAL_DATA_DIR      = _DATA_ROOT / "val_data_alt"
        RESULTS_DIR       = _DATA_ROOT / "results_alt"
    else:
        BASELINE_DATA_DIR = _DATA_ROOT / "baseline_data"
        TEST_DATA_DIR     = _DATA_ROOT / "test_data"
        VAL_DATA_DIR      = _DATA_ROOT / "val_data"
        RESULTS_DIR       = _DATA_ROOT / "results"

    ADD_AUTOPILOT_VEHICLES = True

    FRAMES_TO_SKIP = 0      # 0 -> Every frame is attacked individually
    WOR_EPSILON = 4.0           # 5 -> No effect   # Wor: 8    # TF: 12

    # PGD / FGSM attack settings (TFV6 adversarial perturbation)
    # PGD_TARGET: "brake" | "max_speed" | "steer_left" | "steer_right"
    PGD_TARGET  = "brake"
    PGD_EPSILON = 4.0       # ε budget (pixel units); must match hpc/prep_test.py PGD_EPSILON default
    PGD_N_STEPS = 5         # PGD iterations; converges in ≤2 steps for brake target at ε=4

    DEFAULT_CMD = 2
    SHRINKAGE_ALPHA = 0.01   # covariance shrinkage factor applied at fit time
    MAHAL_RIDGE     = 1e-6   # numerical stabiliser added at score time (not a modelling choice)
    GMM_MAX_K = 10
    GMM_COV_TYPE = "full"
    RANDOM_SEED = 17

    # If True, attention profiles are built from the wide-camera relevance map
    # only.  The narrow-camera contribution is ignored in _give_element_selectivity.
    # Profiles are re-normalized to sum 1 as usual, so all downstream detectors
    # work without modification.
    WIDE_ONLY_PROFILE = True

    # Profile-block flags.  Each enabled block appends num_classes dims to every
    # ATOMs profile (TFV6: 10 per block) at the cost of one extra LRP backward
    # pass per frame.  Blocks are normalized to sum 1/n_blocks each, so the full
    # profile always sums to 1 (valid distribution for JSD/Wasserstein).
    # Block order: default | brake (if on) | waypoint (if on).
    # NOTE: baseline/test/val profiles must be recomputed after toggling either.
    #
    # *** Both DEFAULT OFF after measurement (2026-07-02): for TFV6, every
    # decoder-level seed tested (brake one-hot, wp-query activations, lateral
    # waypoint decision) produces a pixel map with cosine >= 0.994 to the
    # default speed-seeded map — the extra blocks are redundant copies.  See
    # "Seed-invariance of TFV6 LRP maps" in docs/design_decisions.md. ***

    # Brake-counterfactual block: one-hot seed at speed bin 0, output→input LRP
    # ("what evidence does the model see in favor of stopping?").  Output-level
    # seed — mode-3 flavored regardless of MODE_ANALYSIS.
    ADD_BRAKE_SEEDS = False

    # Waypoint-head block: positive-activation seed at the waypoint query tokens
    # ([N_wp, 256] — the wp head's F_c analogue, penultimate before the per-token
    # wp_decoder Linear), backprop to pixels.  The exact mode-2 analogue of the
    # default speed_query seed, applied to the lateral/spatial decision head.
    # TFV6 only.
    ADD_WAYPOINT_SEEDS = False

    # If True, feed the model real target-point conditioning (current/previous/
    # next TP, ego frame) from the frame npz instead of zeros.  Requires frames
    # migrated with a version of migrate_lead_to_baseline.py that stores the
    # target_point* keys (2026-07-02+); older npz files fall back to zeros with
    # a one-time warning.  The deployed TFV6 uses three TP tokens
    # (use_tp/use_previous_tp/use_next_tp), so zeroed TPs mean degenerate route
    # conditioning that shifts attributions relative to the live agent.
    USE_REAL_TARGET_POINTS = True

    # If True, the per-class relevance in _give_element_selectivity is divided by
    # the number of active (non-zero relevance) pixels, giving mean relevance per
    # pixel (as described in Beylier et al.).  If False (default), the raw sum of
    # relevance over the class mask is used instead.
    NORMALIZE_BY_PIXEL_COUNT = False


    if TOWN == "Town07" and not LIVE_PERTURBATION_RECORDING_MODE:
        SPAWN_INDEX = 85
        SPEC_POS = [-200, -150, 7]
        SPEC_ROT = [-19, -90, 0]
    if TOWN == "Town07" and LIVE_PERTURBATION_RECORDING_MODE:
        SPAWN_INDEX = 58            # 89
        SPEC_POS = [-70, -150, 7]
        SPEC_ROT = [-19, 90, 0]
    if TOWN == "Town02":
        SPAWN_INDEX = 97
        SPEC_POS = [143, 108, 7]
        SPEC_ROT = [-19, 0, 0]
    if TOWN == "Town01":
        SPAWN_INDEX = 60
        SPEC_POS = [90, 64, 7]
        SPEC_ROT = [-19, -0, 0]
    if TOWN == "Town04":
        SPAWN_INDEX = 342
        SPEC_POS = [100, -200, 7]
        SPEC_ROT = [-19, -90, 0]
    if TOWN == "Town03":
        SPAWN_INDEX = 164
        SPEC_POS = [-0, 100, 7]
        SPEC_ROT = [-19, -0, 0]
    if TOWN == "Town06":
        SPAWN_INDEX = 366
        SPEC_POS = [-90, 245, 7]
        SPEC_ROT = [-19, -0, 0]
    if TOWN == "Town05":
        if LIVE_PERTURBATION_RECORDING_MODE and PERTURBATION == "pgd":
            SPAWN_INDEX = 272               # 235
            SPEC_POS = [140, -120, 50]        # [30, 203, 10]
            SPEC_ROT = [-50, -20, 0]         # [-20, -0, 0]
        else:
            SPAWN_INDEX = 152
            SPEC_POS = [30, 148, 10]
            SPEC_ROT = [-20, -0, 0]
# --------------------------------------------------------------------------------------
# Global Helpers
# -------------------------------------------------------------------------------------
    image_counter = 0