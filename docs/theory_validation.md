# Independent finite CPU theory and learning-rate validation

This suite isolates reference alignment, fixed-target transfer and finite-step
cost. It does not change production ImageNet/iNaturalist entry points, methods,
policy modes, schedulers or checkpoints. It intentionally runs on CPU, including
on a TPU VM, and downloads nothing. This finite theory experiment is separate
from the production nohup/TPU training protocol.

## Model, target and update

The default fixed dataset has32 examples and3 features. Random features occur
in identical pairs with opposite labels, guaranteeing nonseparability. There
is one parameter vector theta, no bias, and two logits `[0, x@theta]`. F is the
unsigned CE over ALL training examples. The paired construction limits top1
to50% for any deterministic classifier; optimal CE is log(2) at theta0.
Accuracy is recorded but is uninformative about optimization on this deliberately
contradictory dataset. This is not an image-classification benchmark.

Actual updates call existing ClassificationTask, original SGD or JVP DPCO,
Trainer.step and SGDUpdate. L uses the source-batch mean gradient; E uses the
current misclassified subset's mean CE gradient. The error mask/weights are
frozen. All valid contributions can flip, only strictly negative scores flip,
zero scores stay positive, and empty error references fall back to SGD. The
update denominator remains the original valid B. No BN, augmentation, momentum,
L2, policy mixing or new reference is added.

Separate SeedSequence roles generate data, initialization, training and probes.
Training batches are uniform IID draws WITH replacement. All methods/LRs for
one seed share initialization and the full index sequence. Changing probe counts
does not consume training randomness or alter the trajectory. Metadata records
data/fingerprint, RNG roles, parameterization, source versions, dtypes and settings;
summaries contain the ordered-index SHA256.

ClassificationTask retains float32. Independent NumPy float64 logistic formulas
calculate full-data gradients, small dense gradients and oracle residuals. JAX's
global precision is unchanged. These are numerical finite-data calculations, not
symbolic exact arithmetic. Dense gradients are CPU observers only, never written
to disk or connected to ResNet. The method direction is replayed for observation;
its oracle error and the real parameter-transition error are retained.

## Commands

Use the project environment with existing test dependencies. The limited run
actually executed for this implementation is:

```bash
./run_dpco_theory_suite.sh \
  --seeds 5 --methods sgd dpco_e dpco_l \
  --lrs 0.01 0.1 --schedules constant cosine \
  --steps 60 --batch-size 8 \
  --probe-steps 0 5 20 59 --probe-count 128 \
  --probe-lrs 0 0.001 0.01 0.05 0.1 \
  --case-mc-samples 4096 --output results/theory_validation
```

`DPCO_PYTHON=/absolute/path/to/python` overrides the interpreter. Both JAX platform
variables are explicitly CPU. `--dry-run` prints the grid without JAX initialization
or file writes; `--analytic-only` runs cases A/B/C. `--no-probes` or `--probe-count 0`
disables paired/moment sampling; cheap full-data observation and cases still run.
No model checkpoint is saved.

The former proposed multi-seed constant/cosine replication grid is superseded
by the 2026-10-25 theory-driven plan. The finite run above is historical evidence,
not an instruction to repeat a training grid. The minimal scheduler extension also
accepts `--schedules inverse_sqrt` through this CPU interface, but no CPU training
grid is authorized or executed in the scheduler-only task. Real TPU commands use
run_dpco_mechanism_suite.sh; see docs/tpu_mechanisms.md.

Every method gets the same grid/budget. `best_in_finite_grid.json` selects lowest
final unsigned full-training CE per method/seed/schedule, allowing different best
LRs. This is a finite training-objective grid, not an exact optimum or held-out
estimate. Constant/cosine retain their existing behavior; inverse_sqrt is
`eta0/sqrt(completed_updates+1)` with a fixed exponent and no floor/warmup.
A nonzero `--min-lr` requires
a cosine-only grid; its final update uses the floor, possibly0. Finite-budget
cosine does not establish infinite-horizon convergence conditions.

## Three private recorder streams

Each run owns `logs/{run_name}`, `files/{run_name}`, and an empty `runs/{run_name}`
under the chosen output root. The common RunRecorder receives explicit PRIVATE
schemas; global SCHEMAS and ordinary checkpoint cursors remain unchanged. These
CPU experiments use their own grid_results.json, not the production lifecycle TSV.

| File | Meaning |
|---|---|
| theory_train.csv | Real update step=t+1, explicit before=t/after=t+1; full-data CE/top1, oracle CE, actual LR and cumulative LR/LR-squared, h-squared, actual-direction projection/norm, transition residual, cumulative bound/slack and separate training/diagnostic time |
| theory_pairs.csv | Frozen state t and actual source indices; both E/L versus SGD, explicit probe LR, source full CE, nonempty frozen-error CE and fixed full-training CE |
| theory_moments.csv | Independent MC batches at ONE state; long-format means/SEs for h-dot-directions, Delta/Lambda/mismatch, squared norms, energy, gradient components and alignment ratio |

Trajectory method names where the state came from. The reference column names the
counterfactual direction being examined; both E and L can be probed on SGD states.
LR scans never retrain/change the trajectory. Source images and E masks remain
fixed during each scan, even if predictions change. Empty error objectives are
omitted, not assigned fictitious CE; eta0 gives blank H/eta and residual fields.

Delta=r^T(Gs-Gd); Lambda=h^T(Gs-Gd); mismatch=(r-h)^T(Gs-Gd).
Rows retain T1 residual Delta+2/B*sum(min(q,0)), transfer residual
Lambda-Delta+mismatch, and energy slack mean(||g_i||^2)-||Gs||^2.
The paired residual is H/eta+grad(J)^T(Gs-Gd). Only the matching reference
objective identifies this gain with Delta; the full target uses Lambda.
No unproved SGD-style sigma-squared/B bound is imposed on DPCO.

MC standard errors are sample SD/sqrt(n), not proofs. The alignment ratio is
undefined/blank if ||h||^2<=1e-12, with no hidden epsilon. Frozen-state probes
do not establish a uniform positive alignment constant. Second moments are
not relabeled as center variance or error relative to h.

The valid global bound is L=lambda_max(X^T X)/(4N), since the no-bias model's
Hessian is mean(p_i(1-p_i)x_ix_i^T). Adding bias would require new augmented
features and a recalculated bound. On each ACTUAL path the certificate is
F(theta_T)<=F(theta0)-sum(eta*h^T*d)+(L/2)*sum(eta^2*||d||^2).
Pre-update h and d come from that same path. Actual float32 rounding is retained:
raw slack and transition errors are recorded; the pass flag permits the explicit
2e-6*T numerical tolerance. Full-data oracle CE is used for the bound, with task
CE recorded separately. This does not order two independent trajectories.

Train duration measures Trainer.step+sync including compilation; direction replay,
full-target evaluation and probes enter diagnostic duration. Loop/writing overhead
remains in elapsed time. This is not production optimizer throughput. Failed
configurations retain CSV/logs/failure.txt and the rest of the grid continues.
Negative gains and unfavorable LR results are never filtered out.

## Analytic cases and observed results (2026-10-24)

The suite's logs/theory_validation_{timestamp}/analytic_cases.json records:

- **A, T1/T2/trajectory limits:** the prescribed two-quadratic case, theta0=.25,
  eta=.1, updated through existing methods/Trainer. DPCO cycles around +/- .05;
  SGD first has lower trajectory loss at update16. At common theta=.05,
  H=.0002375 matches -eta*Delta+eta^2*(1-theta^2)/2. This is finite-step cost
  with exact reference, not a reference mismatch.
- **B, T1 versus target transfer:** the no-bias logistic pair theta1,z=(-.01,1).
  Delta>0 but h^TGs is about -.0180761. At eta=.01, actual E changes audited F
  from .505710684 to .505891537, whereas SGD reaches .505536647. No loss/mask
  definition was changed to eliminate the counterexample.
- **C, conditional moments:** all9 ordered IID size-two batches from the recorded
  three-example dataset, each probability1/9. Exhaustive float32-kernel moments
  and independently sampled MC (n4096) are labeled separately. Exact SE is blank;
  MC deviations use recorded SEs (six-SE diagnostic check), never exact equality.

The finite run completed all12 jobs,720 real updates. All analytic checks and all
recorded cumulative bounds passed. Final unsigned full-training float32 CE:

| Schedule | Initial LR | SGD | DPCO-E | DPCO-L |
|---|---:|---:|---:|---:|
| constant | .01 | .811682 | .788811 | .794205 |
| constant | .1 | .696025 | .693227 | .711293 |
| cosine to0 | .01 | .837404 | .825191 | .827326 |
| cosine to0 | .1 | .718633 | .693147 | .695481 |

The single-seed tiny-data grid separates T1 gain, a valid T3-type cumulative
inequality and final trajectory ordering. DPCO-L loses in one constant-LR row
and wins in the cosine row; neither is universal. At tiny probe LRs, float32
subtraction can dominate H/eta, and those residuals remain in the output. This
does not establish the cause of the ResNet reversal or global DPCO convergence.

Final verification passed all160 CPU/offline tests, including an injected failing
LR followed by a successful LR in the same grid. The final bounded rerun's suite
is `results/theory_validation/logs/theory_validation_20261024_214727_436403/`;
the earlier completed run is retained rather than overwritten.

Across the finite run, the largest T1 identity residual was4.44e-8, independent
direction-oracle L2 error5.40e-8, and actual parameter-transition L2 error4.21e-8.
Minimum raw sample-energy slack was .2190; minimum raw cumulative-bound slack
over all720 updates was1.8773e-6. All runs shared one training-index fingerprint.
These are numerical checks at the tested states, not global certification.

At the late DPCO-E/cosine/.1 state59, a hypothetical .1 probe step had positive
target correction projection about3.47e-5 but full-target H about1.95e-4>0,
consistent with a finite-step cost overwhelming the small first-order gain.
The ACTUAL final cosine LR was zero; this probe must not be described as its
executed training update. Other L-source probes had Delta>0 but Lambda<0, which
separately illustrates transfer failure on the fixed full target. Repeated LR
scales are not independent conditional expectation samples.
