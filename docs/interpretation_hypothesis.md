# ATOMs Live-Perturbation: Interpretation Hypotheses

This document records interpretations of observed detector behaviour in the live-perturbation
experiments (PGD attack, `run_online_analysis.py`), based on profile-level inspection
of `pgd_brake_205328_000` and `pgd_nocrash_155706_000`.

---

## Observed pattern

The k-NN distance trace has three distinct phases (same shape for both variants):

| Phase | Frames | k-NN distance |
|-------|--------|---------------|
| Pre-injection (clean driving) | 0–39 | ~0.27 (stable) |
| Injection spike | 40–45 | peaks ~0.32–0.35 |
| Post-perturbation crash | 45–end | crashes to ~0.05–0.15 |

---

## Variant-level speed data

Both variants start clean at ~8–9 m/s but diverge sharply at injection:

**`brake` variant** (100 frames):
- Frame 40: injection starts at ~9.3 m/s
- Frame 44: speed = 0.0 m/s — vehicle fully stopped
- Frames 45–99: vehicle remains stopped for the entire remainder of the recording

**`nocrash` variant** (76 frames):
- Frame 40: injection starts at ~10.9 m/s
- Frames 41–47: vehicle *accelerates* from ~11 → 21 m/s
- Frames 47–75: vehicle holds ~20 m/s (maximum speed) for the rest of the recording

These are opposite behaviours: one stops, the other runs at maximum speed.

---

## Root cause: the "stable-but-wrong-state" problem

### 1. ATOMs profile shift — identical in both variants

Pre-injection, both profiles are dominated by **RoadLine** (class 5, ~65–67%) with smaller
contributions from Unlabeled (class 0, ~19–23%) and Road (class 2, ~15–17%).
**Vehicle (class 1) = 0 in all pre-injection frames.**

From ~frame 48–50 onward, both variants show the same shift:
- **Vehicle (class 1) rises from 0 → ~0.27–0.44**
- **RoadLine (class 5) drops from ~0.65 → ~0.33–0.49**

The resulting post-perturbation profile is approximately
`[Unlabeled≈0.14, Vehicle≈0.30, Road≈0.13, RoadLine≈0.41, …]`
in both cases.

### 2. Physical cause differs, but attentional outcome is the same

- **Brake variant:** the vehicle stops behind a lead vehicle; it gradually fills the windshield.
- **Nocrash variant:** the vehicle accelerates to ~20 m/s, catching up with surrounding traffic
  and closing the gap to vehicles ahead.

Both trajectories end at "agent very close to another vehicle" → Vehicle class dominates LRP.

### 3. Why k-NN crashes below the clean-driving level

The profile `[Vehicle≈0.30, RoadLine≈0.40, …]` is the canonical
**"following another vehicle in urban traffic"** attention pattern — one of the most common
and densely populated states in the LEAD baseline dataset (heavy urban driving across six towns).

The k-NN detector (`normalize=True`, k=25) measures distance on the unit sphere after
L2 normalization. When the test point lands inside a dense baseline cluster, k-NN → 0.
A vehicle in close proximity to traffic is indistinguishable from a vehicle *normally* in
close proximity to traffic, so the detector reports near-perfect in-distribution.

This also explains why post-perturbation distances (~0.05–0.15) fall *below* the clean
pre-injection level (~0.27): the live recording is already slightly OOD from the LEAD
baseline when clean (different town, live rendering), but the "vehicle ahead" profile
lands in one of the baseline's densest regions.

---

## Key insight: scope limitation of ATOMs-based OOD detection

ATOMs/k-NN answers: **"does the agent attend to an unusual set of objects right now?"**

It does **not** answer: **"is the agent in an anomalous causal state?"**

A vehicle forced to stop (or to accelerate into traffic) by a PGD attack ends up with an
attention profile that is semantically identical to a vehicle *normally* navigating urban
traffic. The detector correctly identifies the perturbation *at the moment of onset*
(the spike at frame 40), but loses signal once the scene settles into a visually familiar
configuration — regardless of whether that configuration was reached via braking or
runaway acceleration. The footprint of the *consequence* is in-distribution; only the
footprint of the *transition* is OOD.

This is a fundamental temporal-horizon limitation:
- **ATOMs detects perturbation onset** — the transition from one attentional regime to another.
- **ATOMs cannot detect sustained effects** that push the agent into a stable but causally
  wrong state whose LRP footprint matches the baseline.

---

## Implications for the thesis

1. **Offline (single-frame) evaluation is the natural regime for ATOMs.** Each perturbed frame
   is scored independently; the transition signal is always present. This is why offline AUC
   results are stronger.

2. **Live evaluation exposes the time-horizon problem.** The injection spike is detectable,
   but sustained post-injection frames may score as *more* in-distribution than clean frames.
   Reporting detection purely on the moment-of-injection frames would be the honest framing.

3. **To detect sustained adversarial effects, the detector would need temporal context:**
   speed history, trajectory deviation from the planned route, or a model that knows the
   agent *should* be moving. ATOMs alone cannot provide this signal.

4. **The finding itself is thesis-worthy:** it demonstrates that attention-based OOD detection
   is sensitive to the *causal chain* leading to an attentional state, not just the state
   itself — a meaningful distinction for safety-critical systems.

---

## Open questions

- Does the Mahalanobis or MDX detector behave differently in the post-perturbation frames?
  If MDX (backbone features) also crashes, that confirms the backbone representation of
  "vehicle in close proximity" is equally in-distribution. If it stays elevated, that would
  be a useful differentiator.
- Would an ensemble of ATOMs + speed-aware signal (e.g., flag frames where predicted speed
  >> actual speed, or flag anomalous acceleration) recover sustained detection?
- Is the Vehicle-class rise purely a consequence of a physical vehicle entering the camera
  frame, or is it partly an LRP artefact from the changed speed conditioning
  (speed=0 concentrating the seed on bin 0, speed=20 concentrating it on bin 8)?
- Does the `nocrash` variant name refer to the agent not crashing despite running at 20 m/s,
  or to a different PGD objective? Clarify to ensure the variant comparison is meaningful.
