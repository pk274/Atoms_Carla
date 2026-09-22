# Archive

Material that is not used by the thesis, moved here unchanged during the hand-in cleanup
(2026-09-22), so its git history is preserved (`git log --follow archive/...`). Paths
inside these files still refer to their old locations.

The complete pre-cleanup state is the tag `pre-handin-cleanup`:

```bash
git checkout pre-handin-cleanup -- <path>
```

## `hpc_wor/`

The SLURM chains (prep -> array -> gather) for the **World on Rails** agent, which the
project used before switching to TransFuser v6. The thesis reports TFV6 only. The WoR
agent code (`pcla_agents/wor/`) and its LRP implementation
(`ATOMs_Analysis/saliency/lrp_analysis.py`) stay in place as part of the framework.
`docs/cluster_explanations.md` still describes these pipelines.

## `results_summaries/`

Reports written by `summarize_results.py`, all superseded:

* `results_summary_WOR/`: the WoR cluster sweep.
* `results_summary_3_cams_alt/`: TFV6 on the earlier three-camera dataset, which gave
  wrong attributions because TFV6 was trained on six cameras.
* `results_summary_alt_pre_leak_fix/`: the TFV6 alternative-split sweep of 2026-07-17,
  computed before the reference-set leak was fixed on 2026-09-20. The thesis numbers come
  from the recomputed results. Run `python summarize_results.py` to regenerate the report.

## No longer tracked

These were removed from the git index. The local copies were kept where they existed,
and all of them are in the tag.

* `data/WOR/` (1.1 GB): WoR reference and test data.
* `data/TFV6/test_data/`: live-run arrays of the original Town05-held-out split. The
  thesis uses the alternative split (`test_data_alt/`).
* `dataset_example_folder/`: one example LEAD route, not referenced by any code.
* `documents/thesis_style.py`: a byte-identical duplicate of `thesis_style.py`.
