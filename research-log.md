# SHARE Optimization Research Log

## 2026-08-29 — Initialization and evidence audit

- Declared the optimization targets before new runs.
- Preserved all opened independent results as diagnostic-only artifacts.
- Confirmed that the current SHARE result at the 20% cap is
  Repair.SR/HIR/UAR/Stop = 0.862/0.178/0.097/0.085 with SDs
  0.057/0.034/0.044/0.072.
- Confirmed that the prior V3 attempt is stable but not suitable: at 20% it
  obtains 0.912/0.070/0.164/0.000 with SDs 0.010/0.016/0.013/0.000.
- Confirmed the distribution-shift signal: development VoH rho was about 0.285,
  whereas the opened independent post-hoc analysis gives 0.107.
- Next action: reproduce the development baseline and decompose variance before
  changing the model.

## 2026-08-29 — Stability and VoH optimization

- Reproduced the development five-seed paired-replay result. Its development-test
  SDs were Repair 0.0217, UAR 0.0159, and Stop 0.0190, establishing that value
  averaging—not longer training—was the plausible stability intervention.
- Found and fixed a real `DQNEnsemblePolicy` defect: scalar inference averaged
  members, while batch inference used the ensemble object's unused random network.
  Added a regression assertion that scalar and batch ensemble values agree.
- Tested a decision-active realized pairwise-VoH head. It improved ranking on the
  original development test (rho about 0.36--0.41) but harmed Repair/Stop under
  calibration shift, so it was rejected as the primary controller.
- Tested safety-weighted targets, direct human-beneficial labels, augmented-data
  heads, blended DDQN/VoH scores, and worst-domain calibration. None improved the
  three action metrics jointly; all rejection results remain in their separate
  result ledgers.
- Diagnosed the simulator's single-realization noise: branch success is sampled
  from a stable hash of episode/step, whereas the state identifies branch
  probabilities. This attenuates realized-return VoH correlations across groups.
- Froze the existing development-selected five-member arithmetic Q ensemble.
  A diagnostic on the already opened 400-group V2 confirmation improved the old
  SHARE result to Repair/UAR/Stop = 0.880/0.0856/0.070.
- Generated a new 250-group V4 holdout from the 632 remaining public-API groups.
  All five overlap dimensions passed. The one-shot frozen ensemble achieved
  Repair/HIR/UAR/Stop = 0.932/0.200/0.0688/0.044.
- Post-confirmation leave-one-member-out sensitivity achieved SDs 0.0150/0.0116/
  0.0252 for Repair/UAR/Stop, all within the declared targets.
- Trained an expected-VoH head on V1--V3 replay data and selected its fixed
  operating threshold on V4. Expected VoH uses simulator probability metadata;
  no V5 outcome was available during fitting or threshold selection.
- Generated a final, disjoint 150-group V5 holdout from the 382 remaining groups.
  The one-shot expected-VoH result was rho 0.989, CI [0.986, 0.991], with strictly
  increasing quintiles. The realized-return sensitivity was rho 0.175, CI
  [0.082, 0.265], also with strictly increasing quintiles.
- Final interpretation: use the Q ensemble as the primary authority controller.
  Use expected VoH as the simulator-identifiable mechanism score and disclose the
  weaker single-realization sensitivity rather than replacing it silently.
