# Exported result CSV guide and training plots

This guide describes the actual recording code and the ImageNet100/ResNet50
exports inspected on 2026-10-27. It distinguishes real trajectory measurements
from read-only, same-state hypothetical updates. All accuracy values in source
CSVs are fractions. The plotting script displays training accuracy as percent.

## 1. What is currently available

The supplied `../result/probe/` tree contains 36 complete runs of 5,000 updates,
seed 5 and global batch 256: 17 at initial LR 0.05 and 19 at initial LR 0.01.
Each LR has nine constant-rate runs (SGD, E/L continuous, E/L half, E/L
alternating with both first methods), six decay runs (SGD/E/L for cosine and
inverse_sqrt), and two E/L warmup10 runs. LR 0.01 also has two E/L warmup30
runs. Training curves contain 180,000 rows. Warmup is an SGD prefix before
DPCO, not an LR warmup.

Seventeen CSV filenames occur in these exports. The older LR 0.05 decay exports
use `mechanism_*`; the new LR 0.01 decay exports use `diagnostic_*`. These are
different observer schemas, not evidence that training failed to produce data.
Legacy `theorem1.csv`, `theorem2.csv`, `theorem3.csv` and
`mechanism_windows.csv` do not occur in this particular input tree. They are
documented below because other existing project runs can contain them.

Files and logs can be exported into separate `files_*` and `logs_*` directories.
Join them by exact run directory name, not by folder date or a guessed label.
`logs_*/<run_id>/metadata.json` contains the effective configuration, code
commit/content fingerprint, device environment and dataset protocol identity.
The lifecycle start/end table is a TSV under logs, not a per-step metric CSV.

## 2. Common meanings and notation

- `step` in `train_steps` or `policy_steps`: number of the just-completed update.
  Training row k describes the batch prediction at theta_(k-1), before update k.
- `step` in `diagnostic_*` or `mechanism_*`: completed updates BEFORE the inspected
  update. `update_step = step + 1` matches the training row. For example,
  diagnostic state 25 checks training update 26. Never join these using step
  alone. Legacy theorem streams have a separate `phase` convention.
- Gd is the unmodified valid-batch mean gradient. Gs is FULL DPCO's flip direction.
  C = Gs - Gd. A policy direction is G_alpha = Gd + alpha*C.
- Delta = r^T C is the reference projection gain. It is not a loss difference.
- For any frozen objective J, gain_J = grad J(theta)^T C.
  For a fixed training target F_A, call this projection Lambda.
- H(eta) = J(theta-eta*Gs) - J(theta-eta*Gd). H<0 means a relative DPCO advantage.
  It does not alone mean either candidate reduced J below its initial value.
- The first-order residual is H/eta + gain_J. It is undefined at eta=0 and is
  stored blank, not as a successful zero. A tiny H can be limited by loss precision.
- E's own objective is CE over the frozen, pre-update misclassified subset.
  L's own objective is the entire realized source batch's CE. They are not the
  same objective, and neither automatically equals a fixed dataset-wide target.
- Masks, inputs, augmentation and input BN buffers are frozen for a candidate
  comparison. These are train-mode whole-batch objectives unless explicitly
  labeled otherwise; normal validation uses running-BN eval mode.
- `trajectory_method` identifies the method whose training state is observed.
  `reference` identifies the candidate reference being examined. A row on an SGD
  trajectory can still describe a hypothetical FULL DPCO candidate.
- `norm_sgd_sq` and `norm_dpco_sq` are squared gradient-direction norms. They
  are not centered variances. In policy diagnostics, the latter remains the full
  DPCO counterfactual norm, not necessarily the actual half/SGD-round norm.

## 3. Ordinary training files

### train_steps.csv: the real training curve

One row for every real parameter update. No additional evaluation dataset is
traversed to produce this row.

| Field | Meaning |
|---|---|
| step | Update row k; the metrics were calculated before this update |
| samples_seen | Cumulative valid training examples processed, including this batch |
| epoch_progress | samples_seen / configured training sample count |
| learning_rate | Actual LR used by this update, after schedule evaluation |
| valid_batch_size | Valid contributions in the original batch denominator |
| train_ce | Original unsigned mean CE of the current augmented batch |
| train_top1 | Fraction of current-batch predictions matching the true label |
| elapsed_wall_s | Recorded host-side elapsed run time at dispatch; use synchronized timing windows for throughput |

The loss is not multiplied by DPCO's signs before logging. These are online
batch metrics, not a full training-set re-evaluation at every step. Different
methods see their own model's predictions even under a shared input protocol.
To describe late training, use a declared interval such as rows 4001-5000;
`train_steps` does not evaluate the final post-update theta_5000 on the full set.

### policy_steps.csv: what the real method executed

| Field | Meaning |
|---|---|
| step | Completed real update number |
| policy_mode | continuous, half, alternating, or warmup |
| policy_alpha | 1 for full DPCO; 0.5 for half/filter; 0 for SGD |
| executed_method | dpco, half_dpco, or sgd |
| reference | error or loss; SGD policy rounds do not use it for alignment |

For full-valid-batch algebra, alpha=0.5 removes negative-alignment contributions
while retaining positive contributions. It does not halve the whole full-DPCO
update. Alternating first-method choice is stored in metadata and reflected in
this sequence. Original plain SGD has no policy file. Raw `dpco` without a policy
or warmup wrapper need not have one either.

### evaluation.csv: held-out validation, not the training curve

| Field | Meaning |
|---|---|
| step, samples_seen, epoch_progress | Position of the evaluated model AFTER these updates |
| eval_sample_count | Number of valid images actually evaluated |
| val_ce | Mean unsigned CE over the configured validation split |
| val_top1, val_top5 | Validation accuracy fractions |
| elapsed_wall_s | Elapsed run time when evaluation completed |
| eval_duration_s | This evaluation's wall-clock duration |

Initial step 0 and final evaluation are included. This is useful for model
quality/generalization reporting, but is separate from the professor's current
training-objective-first T1-T3 investigation. Do not label validation as train
accuracy, or select the final test result from training-batch accuracy.

### summary.csv: terminal run status and final validation summary

| Field | Meaning |
|---|---|
| status | completed or failed |
| step, samples_seen, elapsed_wall_s | Terminal progress and elapsed run time |
| final_val_ce, final_val_top1, final_val_top5 | Last validation result at the fixed budget |
| best_val_top1, best_step | Best observed validation top-1 and its model step |
| error | Failure description when present |

This file does not contain final training CE, training accuracy or gradient
variance. `completed` is not numerical diagnostic qualification or theorem
acceptance. The last train_steps row also describes the pre-final-update batch.

### timing.csv: costs and resources

This is an event table; different events populate different fields. Blank
fields are not measured zeros.

| Field | Meaning |
|---|---|
| step | Completed-update position of the timing event |
| event | compile_warmup, train_window, evaluation, diagnostic, probe, checkpoint, memory_snapshot, etc. |
| duration_s | Duration of the named event |
| window_steps, window_samples | Work in a synchronized training timing window |
| steps_per_s, samples_per_s | Throughput of that window, not elapsed time divided by step |
| elapsed_wall_s | Run clock at the event |
| checkpoint_bytes | Snapshot bytes for a checkpoint event |
| device | Device whose memory was queried |
| peak_device_memory_bytes | Peak memory reported by that device, when supported |

DPCO's required reference/scoring/weighted-gradient operations belong to training
cost. Diagnostic overhead and compiler warmup must not be described as ordinary
steady-state SGD/DPCO per-step time.

## 4. Real-update diagnostic files

All five diagnostic CSVs share: `step`, `update_step`, `source_batch_id`,
`trajectory_method`, `reference`. Multiple objectives, LR scales and audit
batches produce several rows per frozen state. They are not independent seeds.

### diagnostic_context.csv: identity and coordinate information

| Additional field | Meaning |
|---|---|
| learning_rate | LR of the actual upcoming update |
| cumulative_eta_before | Sum of actual LRs before the inspected update |
| objective_mode | Train/eval and input-buffer convention for the diagnostic |
| source_context_json | Exact source indices, epoch/offset, augmentation RNG protocol and data identity when supplied |

This supports comparing input protocols and identifying the inspected batch.
Equal seed alone is not a verified image-array, initial-parameter or RNG hash.
`cumulative_eta_before` is not always `initial_lr * step` under decay.

### diagnostic_update.csv: T1 candidates and implementation consistency

| Additional field | Meaning |
|---|---|
| learning_rate, valid_batch_size | Actual LR and original valid B |
| error_count | Current valid samples predicted incorrectly |
| flip_count | Number of negative FULL DPCO candidate weights, not necessarily executed policy flips |
| reference_valid | Whether the selected reference subset is nonempty |
| reference_delta | FULL DPCO candidate r^T C, even on a half/SGD-round trajectory |
| identity_residual | r^T C minus the independently summarized score/weight identity |
| score_min_abs | Smallest absolute valid score; near-zero scores can have fragile signs |
| norm_sgd_sq, norm_dpco_sq | Squared norms of full SGD and full DPCO candidates |
| direction_l2_error | L2 difference between standalone method replay and independently constructed active direction |
| direction_relative_error | direction_l2_error divided by independent active-direction norm |
| parameter_l2_error | Actual after-parameters minus before-parameters updated with the independent active direction |
| parameter_relative_error | Parameter error divided by independent expected update displacement norm |
| replayed_parameter_l2_error | Actual after-parameters minus before-parameters updated with the replayed method direction |
| model_state_max_abs_error | Largest actual-versus-independent output model-state discrepancy |
| replayed_model_state_max_abs_error | Largest replay-versus-independent output model-state discrepancy |
| finite | All checked error values finite; not a tolerance-based qualification |

Relative errors are fractions: 0.07 means 7%. The parameter denominator is one
update's displacement, not the whole model's parameter norm. This diagnostic
file has no `reliable` column. For policy T1 evidence, join `update_step` to
`policy_steps.step`; actual reference gain is alpha*reference_delta. Do not call
alpha=0 rounds positive DPCO interventions.

### diagnostic_paired.csv: FULL DPCO versus SGD on frozen source objectives

| Additional field | Meaning |
|---|---|
| objective_mode | Objective's train/BN/buffer semantics |
| learning_rate | Training LR at that state |
| reference_delta | FULL DPCO reference gain; not automatically this objective's gain |
| norm_sgd_sq, norm_dpco_sq | Full candidate direction energies |
| selected_count | Number of images included in this objective |
| lr_scale | Multiplier applied to the training LR for this candidate comparison |
| probe_lr | Actual candidate step size: learning_rate*lr_scale |
| j_initial | Objective at the common pre-update parameters |
| j_sgd, j_dpco | Objective after the two candidate updates |
| difference | H = j_dpco - j_sgd; negative is a relative full-DPCO win |
| difference_over_lr | H/probe_lr; blank when probe_lr=0 |
| objective_gain | grad(objective)^T C, the correct first-order projection for this objective |
| first_order_residual | H/probe_lr + objective_gain; blank at zero LR |
| actual_update_in_pair | Whether one endpoint is the genuine training result at this LR |
| finite | Candidate values and projections are finite |
| objective | full_batch_ce or frozen_error_ce |

For DPCO-E, frozen_error_ce is the direct reference target; full_batch_ce tests
transfer to the whole source batch. For DPCO-L, full_batch_ce is the direct
reference target. The error set is fixed before both candidate updates.
At scale 1, a matching full-DPCO or SGD executed endpoint is used. Half rounds
do not insert their parameters as a full-DPCO endpoint.

### diagnostic_transfer.csv: FULL DPCO candidates on other training batches

All paired fields above apply, except that the objective is a fixed-member
training audit batch rather than the source. Additional fields:

| Field | Meaning |
|---|---|
| target_batch_id | Identity of the audit training batch |
| overlap_sample_count | Source/audit sample-index overlap, when known |
| transfer_difference | objective_gain - reference_delta |

The audit images/augmentations are fixed, but this older diagnostic uses the
current state's before-BN buffers. Do not automatically concatenate it as one
fixed differentiable objective over time. It is not the validation dataset or
the full ImageNet training objective. Negative objective_gain can preserve a
relative disadvantage as probe LR is reduced, despite positive source Delta.

### diagnostic_policy.csv: the ACTIVE policy versus SGD

Use this file for half, alternating and warmup conclusions about the actual
policy, rather than substituting FULL DPCO's paired/transfer rows.

| Additional field | Meaning |
|---|---|
| policy_mode, policy_alpha | Active policy and alpha at this pre-update state |
| objective, objective_mode | full_batch_ce, frozen_error_ce or training_audit_ce, with buffer semantics |
| target_batch_id | Source or audit batch identity |
| lr_scale, probe_lr | Candidate step-size scale and value |
| j_initial | Frozen objective before update |
| j_sgd | Objective after the SGD candidate |
| j_policy | Objective after the active policy candidate |
| difference | j_policy - j_sgd |
| objective_gain | alpha * grad(objective)^T C |
| first_order_residual | difference/probe_lr + objective_gain |
| actual_update_in_pair | Scale 1 uses the real policy endpoint |
| finite | Checked values are finite |

SGD rounds have alpha=0; their theoretical relative policy gain is zero. A tiny
nonzero numerical difference between separately evaluated endpoints is not
DPCO evidence. Each half round has alpha=0.5, not a reduced global LR.

## 5. Extended mechanism CSV files

These currently appear in the six older LR 0.05 decay exports. Their common
columns are `step`, `update_step`, `trajectory_method`, `trajectory_reference`,
`source_batch_id`, `target_id`. `trajectory_reference` describes the real method;
`reference` in candidate rows can instead be loss, error or the exact-bank oracle.
The same training state can therefore have multiple hypothetical reference rows.

### mechanism_context.csv

`learning_rate`, `cumulative_eta_before` and `source_context_json` have the
context meanings above. `target_id` identifies the frozen whole-batch training
bank objective F_A, including its asset/buffer identity, when one is enabled.

### mechanism_update.csv

The replay/independent errors have the meanings in diagnostic_update. Additional
or differently normalized fields:

| Field | Meaning |
|---|---|
| direction_norm | Norm of the independent expected active direction |
| actual_displacement_norm | Norm of actual after.params-before.params |
| parameter_relative_error | Here the denominator is ACTUAL displacement norm; diagnostic_update instead uses expected displacement |
| parameter_dtype, compute_dtype | Stored parameter and model computation dtypes |
| matmul_precision | JAX default matmul precision setting |
| absolute_tolerance, relative_tolerance | Recorded provisional numerical comparison thresholds |
| tolerance_supported | Whether the dtype is covered by the provisional qualification rule |
| reliable | All supported-dtype, finite, direction/update/model-state checks passed their numerical gates |
| finite | Values finite; weaker than reliable |

`reliable` is not an optimizer win or theorem proof. Error magnitudes and which
gate failed must remain visible. It is not acceptable to fix a failed check by
renaming it successful or silently widening tolerances to a few percent.

### mechanism_t1.csv

| Field | Meaning |
|---|---|
| reference, reference_valid | Candidate reference type and whether defined by a nonempty subset |
| error_count | Source misclassified count |
| flip_count, negative_count | Full-candidate flipped/negative-score counts |
| reference_delta | r^T(Gs-Gd) |
| delta_formula | -2/B * sum of negative scores |
| identity_residual | reference_delta - delta_formula |
| score_min_abs | Minimum valid absolute score |
| norm_sgd_sq, norm_dpco_sq | Full direction energies |

This directly measures the T1 algebra for the labeled reference. On an SGD
trajectory or an oracle row it describes a hypothetical DPCO direction.

### mechanism_transfer.csv

| Field | Meaning |
|---|---|
| reference | Candidate reference type |
| scope | Ordinary-source-to-fixed-proxy scope label |
| reference_delta | Delta = r^T C |
| target_lambda | Lambda = h_A^T C, with h_A=grad F_A |
| mismatch | (r-h_A)^T C |
| transfer_residual | Lambda-Delta+mismatch; should be numerically small |
| h_norm_sq | Squared norm of h_A |

This checks the identity Lambda=Delta-mismatch. Lambda<0 means the correction
is relatively adverse to F_A. It does not necessarily mean the COMPLETE Gs
direction has h_A^T Gs<0. That distinction requires the full-direction projection.

### mechanism_pairs.csv

The j_initial/j_sgd/j_dpco/difference/projection/residual meanings are the same
as above, with the following explicit distinctions:

| Field | Meaning |
|---|---|
| reference | loss, error, or exact_bank_reference_oracle |
| objective | source_full_batch_ce, source_frozen_error_ce, or fixed_training_bank_ce |
| buffer_semantics | Source before-state buffers or the bank's fixed input buffers |
| candidate_kind | independent_pair or actual_pair |
| actual_endpoint | none, sgd, or dpco; identifies the genuine executed endpoint |
| lr_kind | absolute common LR or relative multiplier of the training LR |
| lr_scale | Relative multiplier, blank for absolute LR rows |
| probe_lr | SGD candidate LR |
| dpco_probe_lr | DPCO candidate LR, different when norm matched |
| normalization | original or norm_matched, both read-only candidates |
| defined, undefined_reason | Whether the candidate comparison is defined and why not |
| reference_valid | Nonempty reference support |
| objective_gain | Original unnormalized grad(objective)^T C |
| effective_objective_gain | Projection for the actual compared directions, including norm matching |
| difference_over_lr | H divided by SGD probe_lr |
| first_order_residual | difference_over_lr + effective_objective_gain |
| endpoint_loss_error | Actual endpoint loss versus the corresponding independent endpoint loss |
| loss_resolution | Provisional objective-value resolution scale |
| numeric_checked | Whether an update consistency check was performed |
| update_reliable | Whether that consistency check passed |
| signal_reliable | Consistency and loss-signal/endpoint-contamination gates passed |
| finite | Candidate values finite |

To inspect actual-rate own-reference raw T2, filter matching reference, original
normalization, actual_pair, relative LR kind and lr_scale=1; exclude zero LR.
For full hypothetical LR sweeps, select independent_pair and absolute LR rows.
Do not count overlapping absolute/relative labels as independent experiments.
An exact-bank-reference oracle is an observer candidate, not a trained algorithm.

### mechanism_moments.csv

These moments enumerate a uniform distribution over K FIXED WHOLE BANK BATCHES,
at one frozen parameter state and fixed input buffers. They do not estimate the
full ImageNet batch distribution or require transformed samples to be IID.

| Field | Meaning |
|---|---|
| reference | Candidate reference rule used for each bank batch |
| estimator | exact_finite_bank |
| bank_batch_count | K, the number of whole bank batches |
| probability_sum | Sum of enumerated batch-index probabilities |
| standard_error | Blank for exact enumeration; not an estimated population error bar |
| mean_h_dot_dpco, mean_h_dot_sgd | Finite-bank mean h_A^T Gs/Gd |
| mean_delta | Finite-bank mean reference gain |
| mean_lambda | Finite-bank mean h_A^T C |
| mean_mismatch | Finite-bank mean (r-h_A)^T C |
| mean_norm_dpco_sq, mean_norm_sgd_sq | Finite-bank RAW second moments, not centered variances |
| mean_gd_h_l2_error, mean_gd_h_relative_error | Difference between enumerated mean Gd and the same bank's h_A |
| h_norm_sq | Squared norm of h_A |

A negative ordinary-source-to-bank Lambda and a positive enumerated bank mean
Lambda refer to different source distributions. Do not pool them or replace a
uniform theoretical alignment condition by a few empirical bank ratios.

### mechanism_timing.csv

`step` locates the frozen state; `event` distinguishes compilation, execution,
diagnostic compute and recording I/O; `kernel` names the stage; `duration_s`
records its measured duration. These are observation costs, not ordinary
training iteration times.

## 6. Legacy/optional theorem files not present in this export

### theorem1.csv

Shares `step`, `phase`, `probe_id`; `phase=train` denotes actual update-associated
records, while `phase=node` denotes separate fixed-batch counterfactuals.
Fields valid_batch_size/error_count/negative_count/flip_count/reference_valid
describe the batch; delta is r^T C; eta_delta is training_eta*delta when recorded;
reference_projection_sgd/dpco are r^T Gd/Gs; delta_formula and identity_residual
cross-check T1; norm_sgd_sq/norm_dpco_sq are direction energies. Some fields are
blank outside the phase that measures them. These files are not generated when
legacy probes are disabled.

### theorem2.csv

Includes step/phase/probe_id; objective/objective_mode/batch_id/reference_name
identify the frozen objective; selected_count counts its selected images;
learning_rate/lr_scale/probe_lr define the candidate step; j_initial/j_sgd/j_dpco
and difference/difference_over_lr measure paired candidates; delta is the
projection gain for the LABELED objective (reference Delta for frozen-error E,
but full-source grad J^T C for full_batch_ce). first_order_residual checks that
objective's first-order prediction; norm_sgd_sq/norm_dpco_sq record energy;
finite checks numbers and strict_gain_eligible checks positive objective gain
and positive probe LR. Eligibility
does not certify smoothness, a finite-step win or numerical consistency.

### theorem3.csv

Includes step/phase/probe_id/objective/objective_mode. Probe fields audit_loss,
audit_gradient_norm_sq, target_gain, transfer_difference describe the labeled
audit function and correction. delta is reference gain; sample_energy is mean
per-sample squared gradient norm; norm_sgd_sq/norm_dpco_sq are batch direction
energies. Training fields cumulative_reference_gain/cumulative_eta/
cumulative_eta_sq accumulate eta*Delta, eta and eta^2 along ONE trajectory.
They are ingredients or diagnostics, not a proof that independently trained
DPCO must beat independently trained SGD after T steps.

### mechanism_windows.csv

This optional continuous-window stream is absent from the current 36 runs.
It has common mechanism identity columns plus:

| Field | Meaning |
|---|---|
| window_start, window_end | Half-open interval of pre-update states [start,end) |
| window_count, window_complete | Number of observed consecutive updates and whether the whole interval completed |
| fa_before, fa_after | Same fixed F_A evaluated before/after the real update |
| P | -h_A^T u, where u=actual after.params-before.params |
| R | fa_after-fa_before+P, the observed nonlinear remainder |
| displacement_norm_sq | Squared norm of u |
| cumulative_P, cumulative_R | Prefix sums within this window |
| endpoint_loss_difference | Current F_A minus the window's initial F_A |
| identity_residual | endpoint_loss_difference+cumulative_P-cumulative_R |
| identity_tolerance, identity_pass | Recorded numerical telescope check |
| learning_rate, cumulative_eta, cumulative_eta_sq | Actual step size and window sums |

The identity is Delta F_A=-sum P+sum R. A numeric identity pass is not a
convergence certificate or a certified bound on smoothness. Discontinuous or
failed windows must not be silently stitched into a valid cumulative target.

## 7. Mapping to the professor's Theorems 1-3

| Question | Primary evidence | What it does not establish |
|---|---|---|
| T1: does the signed aggregation gain match the reference identity? | mechanism_t1 or diagnostic_update; policy gain needs alpha | Fixed-target benefit, small variance or long-run superiority |
| T2: at the same state, does the candidate improve the matching objective relative to SGD? | diagnostic_paired for full DPCO; diagnostic_policy for active policies; mechanism_pairs for explicit target/LR/numerical semantics | An independent-trajectory comparison or an automatic whole-dataset benefit from E-error benefit |
| Target transfer needed for practical T3 | diagnostic_transfer or mechanism_transfer, with target/buffer identity respected | A population expectation from a tiny audit bank |
| Conditional quality/energy ingredients | mechanism_moments over its explicitly finite distribution | Uniform ResNet alignment/energy assumptions, IID sampling or centered variance estimates |
| Fixed-target accumulated change | optional mechanism_windows; legacy theorem3 with its stated target/phase | Full ImageNet convergence or cross-trajectory ordering |
| Empirical training convergence | train_steps curves, fixed-budget/declared-interval CE and top-1 | A mathematical proof of T3 |
| Final held-out model quality | evaluation and summary | The training-objective-first theorem test |

The original T3's fixed-reference/smoothness/quality assumptions and its
single-trajectory scope remain necessary. Positive Delta at many nodes is
not alone sufficient to conclude a smaller final training loss than SGD.

## 8. Run the plotting script on a Mac

Use an independent local plotting environment if matplotlib is unavailable:

```bash
cd "/Users/jason/Documents/1. Education/9. Codex/Project_DPCO/DPCO"
python3 -m venv ../.venv_plots
source ../.venv_plots/bin/activate
python -m pip install matplotlib

python scripts/plot_training_results.py \
  --input ../result/probe \
  --output ../reports/training_curves_20261027 \
  --smooth-window 50 \
  --formats png pdf svg
```

Default figures each contain train CE and train top-1 panels:

1. continuous: SGD/E/L, constant LR, both initial LRs.
2. half: SGD versus E/L half/filter, constant LR, both LRs.
3. alternating_sgd_first: SGD versus E/L alternating, SGD first.
4. alternating_dpco_first: SGD versus E/L alternating, DPCO first.
5. cosine: SGD/E/L under cosine-to-zero, both initial LRs.
6. inverse_sqrt: SGD/E/L under inverse-square-root decay, both initial LRs.
7. warmup10: E/L with a 10% SGD prefix at both LRs, plus the original SGD
   baseline at each matching LR. Line style encodes initial LR.
8. warmup_percent: E/L with a 10% versus 30% SGD prefix at fixed LR 0.01,
   plus one original SGD baseline at that LR. Line style encodes percentage.

Blue is SGD, red is DPCO-E, orange is DPCO-L. With the default two rates,
initial LR 0.01 uses sparse (8,5) dashes with a 20% lighter method color, and
0.05 uses the solid base color. Line width defaults to 1.05 points. Each panel's
physical width:height is 1:0.8, set using box aspect rather than data-unit aspect.
Figures contain only axes, ticks, axis labels and legend; dataset/protocol,
smoothing, target scope and comparability notes remain in the gallery/manifest.
Use --panel-height-ratio, --line-width and --dash-lighten to control these styles.
Each schedule has common y limits
across its policy figures. Initial LR labels for decay are not constant actual
LRs; actual per-step values are validated against the recorded schedule.

Smoothing is the same causal trailing 50-row arithmetic mean for every curve.
The first 49 positions use a growing prefix; no initial observations are hidden.
All current runs have the same full batch size. Accuracy is multiplied by 100
only for display. No seed aggregation, uncertainty band or best-run selection
is performed. Use `--smooth-window 1` for raw curves. For an early view:

```bash
python scripts/plot_training_results.py --input ../result/probe \
  --output ../reports/training_curves_early_20261027 \
  --groups continuous half --smooth-window 10 --max-step 200
```

To generate only the two warmup comparisons:

```bash
python scripts/plot_training_results.py --input ../result/probe \
  --output ../reports/training_curves_warmup_20261027 \
  --groups warmup10 warmup_percent --smooth-window 50 --formats png pdf svg
```

`--warmup-comparison-lr` defaults to 0.01 and must also be included in `--lrs`.
`--warmup-percents` defaults to `10 30`. In `warmup_percent`, pale sparse
dashes mean 10% and solid curves mean 30%; the SGD baseline stays solid blue.
The default comparison contains five curves, not an LR/percentage cross-product.
The selected fixed LR and style dimension are explicit in the gallery/manifest.
There are currently no 30%-prefix LR 0.05 runs; selecting that rate reports
the missing curves rather than substituting another run. At 5,000 updates,
10% is 500 SGD updates and 30% is 1,500, followed by DPCO.

Other supported x axes
are `samples_seen`, `elapsed_wall_s`, and `cumulative_eta_before`; the last uses
the sum of actual recorded rates, not initial_lr*step under decay.

Outputs: PNG/PDF/SVG figures, `index.html`, `run_inventory.csv`,
`training_summary.csv`, `csv_inventory.csv`, `csv_schemas.json`, and
`plot_manifest.json`. The manifest contains source paths/hashes, selected IDs,
comparison warnings and exact plotting arguments. The summary interval `all`,
`last100` or `last1000` is explicitly recorded; none is labeled a full-set
final training evaluation.

Runs are matched by effective metadata. Repeated comparable runs cause an
error instead of selecting the latest/best; use `--run-ids` to disambiguate.
Missing curves are disclosed, never replaced with zero or another schedule.
Different code revisions across LR cohorts are shown in gallery notes and the
manifest. Equal seed does not by itself prove identical initial arrays or
bitwise-identical inputs/augmentation. Making the panels more square does not
change the curves' ordering or give the first few updates a larger share of a
5,000-step axis. Use --max-step 200 (or a shorter explicitly reported prefix)
for early-stage inspection. Keep the smoothing rule fixed when comparing styles;
do not choose a window or axis range to manufacture an expected advantage.
Plotting preserves the raw files and
rechecks their hashes; it imports no JAX, model, method or trainer code.

Sources of schema definitions: `src/training/recording.py`,
`src/probes/tpu_mechanisms.py`, `src/probes/training_diagnostics.py`,
`src/probes/classification.py` and the matching exported metadata/CSV headers.
