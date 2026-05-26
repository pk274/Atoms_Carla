class ExperimentConfig:
    """
    Central configuration for ATOMs experiments.
    """

    from pathlib import Path

    # Active agent — controls which data subfolder is used.
    # Accepted values: "WOR" (World on Rails) | "LBC" (Learning by Cheating) | "TFV6" (TransFuser v6)
    AGENT = "TFV6"

    TOWN = "Town02"
    WEATHER = "sunny"
    SPEED_MODE = False
    HIGH_SPEED_MODE = False

    BASELINE_RECORDING_MODE = True
    TESTSET_RECORDING_MODE = False
    LIVE_PERTURBATION_RECORDING_MODE = False


    NOISE_INTENSITY = 21        # 25 for day, 21 by night
    BRIGHTNESS_INTENSITY = 4

    PERTURBATION = "phantom_obstacle"
    INTENSITY = 0.08
    INJECTION_TIME = 10            # 10 for live perturbation
    AFFECT_BOTH_CAMS = True
    CAM_INDEX = None               # None for all cams
    MANUAL_SPAWNS = False

    RECOMPUTE_BASELINE = False
    RECOMPUTE_TEST_ATOMS = False
    REAPPLY_PERTURBATIONS = False
    RECOMPUTE_MDX_BASELINE = False

    PLOT_SEG_AND_REL = True
    PLOT_COMPARATIVE_REL = False
    PLOT_INTERVAL = 20           # 50


    IMAGE_SAMPLE_INTERVAL = 25   # 25
    TEST_SAMPLE_INTERVAL = 3        # 11
    MAX_BASELINE_SIZE = 100      # 100
    MAX_TEST_SIZE = 200
    MAX_LIVE_PERT_SIZE = 100
    _DATA_ROOT = Path("C:/Users/paulk/Desktop/Unistuff/Masterarbeit/Code/PCLA/data") / AGENT
    BASELINE_DATA_DIR = _DATA_ROOT / "baseline_data"
    TEST_DATA_DIR = _DATA_ROOT / "test_data"
    RESULTS_DIR = _DATA_ROOT / "results"

    ADD_AUTOPILOT_VEHICLES = True

    FRAMES_TO_SKIP = 0      # 0 -> Every frame is attacked individually
    EPSILON = 8.0           # 5 -> No effect

    DEFAULT_CMD = 3
    MAHAL_RIDGE = 0.01
    GMM_MAX_K = 4
    GMM_COV_TYPE = "full"
    RANDOM_SEED = 17

    MODE_ANALYSIS = 2
    FC_RELEVANCE_FILTER = 0.9

    # If True, attention profiles are built from the wide-camera relevance map
    # only.  The narrow-camera contribution is ignored in _give_element_selectivity.
    # Profiles are re-normalized to sum 1 as usual, so all downstream detectors
    # work without modification.
    WIDE_ONLY_PROFILE = True


    if TOWN == "Town07":
        #SPAWN_INDEX = 62
        #SPAWN_INDEX = 2
        SPAWN_INDEX = 85
        SPEC_POS = [-200, -150, 7]
        SPEC_ROT = [-19, -90, 0]
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
        if LIVE_PERTURBATION_RECORDING_MODE:
            SPAWN_INDEX = 235
            SPEC_POS = [30, 203, 10]
            SPEC_ROT = [-20, -0, 0]
        else:
            SPAWN_INDEX = 152
            SPEC_POS = [30, 148, 10]
            SPEC_ROT = [-20, -0, 0]
# --------------------------------------------------------------------------------------
# Global Helpers
# -------------------------------------------------------------------------------------
    image_counter = 0