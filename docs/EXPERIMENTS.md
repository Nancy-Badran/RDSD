# Experimental protocol

## Scope

This is a controlled synthetic evaluation of the PRS computation module, not a
claim of empirical harm prediction or a completed SSI wallet. It extends the
existing binary BN demo without changing its default 10,000-profile generation.

The experiment configuration supplies a monotone OR-of-AND policy over six
attribute names, available credential fields, and one hypothetical holder who
satisfies all three eligibility clauses. Exhaustive subset enumeration returns
37 compliant sets, including three minimal pairs. Only policy-referenced
attributes are enumerated: it is not an exhaustive search over all 20 variables.
It does not implement or claim to reproduce PFA/SMT or Saha et al.'s full system.
The compared sets contain raw category values, not predicate or zero-knowledge
proof outputs; no VP is constructed.

## Questions and artifacts

| Question | Measurement | Artifact |
|---|---|---|
| Does equal cardinality imply equal leakage? | Per-target normalized leakage and PRS of the three minimal pairs | scores.csv, results_table.tex, leakage.pdf |
| Does dependency-aware selection differ from baselines? | Reference-model risk and regret of selected candidates | baselines.csv |
| How does history affect additional leakage? | Conditional MI for empty history, AgeBand, AgeBand+Region | scenario_summary.csv, history_weights.pdf |
| How sensitive are scores to impact assumptions? | Equal, income-priority, health-priority normalized weights | scenario_summary.csv, history_weights.pdf |
| What does threshold filtering retain? | Accepted count and first ranked candidate at each threshold | thresholds.csv |
| How does sample size affect learning? | Held-out NLL, edge recovery, leakage MAE relative to reference | validation.csv, leakage_accuracy.csv, sample_size.pdf |
| What is the measured computational cost? | BN training and scoring times, with cache policy disclosed | validation.csv, timings.csv |

Default: three generation seeds, nested training sizes 1,000 / 4,000 / 8,000,
2,000 disjoint test profiles per seed, 20 binary variables, ordered BIC search,
maximum two parents, Laplace pseudocount one. The exact reference BN is used
only for generation and evaluation, never for fitting the learned model. Training
and test samples share the same generating distribution. This is a strong-effect
scenario with a supplied ordering; results are not evidence of distribution-shift
robustness. Error bars are sample standard deviations, not confidence intervals.

## Leakage and history semantics

The scorer computes

    leakage(s,D | O) = [H(s,O)-H(O)-H(s,D,O)+H(D,O)] / [H(s,O)-H(O)]

using exact marginalization of the BN's local CPT products. It enumerates the
ancestors of queried variables only, caches entropy queries, and fails clearly
when their joint state space exceeds its configured bound (default 2^22).
CPTs may have finite multivalued domains; the existing data generator and learner
remain binary. The procedure is exponential, not a scalable production engine.

O is a set of already revealed attribute *variables*. Conditional MI averages
across possible O values; it does not condition on Alice's actual observed
values. It also assumes the same history is known to the assessing adversary.
No cross-verifier linkage, history acquisition, Bayesian-network update, or
population learning from a single holder's events is implemented. These require
separate data and adversary assumptions. Updating O after a presentation means
union with the selected disclosure, not retraining the BN.

If residual entropy is at most 1e-12 bits, the scorer assigns zero *additional*
leakage and flags that target as saturated. This explicit extension avoids an
undefined 0/0 ratio. It does not restore privacy or remove that target's weight;
a low incremental score can coexist with extensive prior exposure. The output
records residual entropies and saturation flags for audit. Thresholds consequently
apply to incremental normalized PRS, not accumulated total disclosure risk.

## Impact weights and baselines

No ITAP records are used. The runner reads hypothetical nonnegative comparable
impact magnitudes from config and normalizes them. Equal (1,1,1), income-priority
(8,1,1), and health-priority (1,1,8) are sensitivity scenarios, not empirical harm
estimates. Missing, negative, nonfinite, and all-zero magnitudes are rejected.
An empirical EstimateImpact procedure is not implemented in the absence of data.
Normalizing heterogeneous monetary and nonmonetary units is not sufficient to
make them comparable.

Baselines all receive the same feasible candidate sets:
- Minimum cardinality: chooses the fewest disclosed fields.
- Additive attribute cost: minimizes configured public-field costs. Costs are
  all one by default, so this baseline equals minimum cardinality in this setup.
- Independent BN: estimates marginals only, removing dependency information.
- Uniform random: exact expected score over all candidates, with no sampling noise.

Selection is evaluated with the known reference BN using the same history and
impact scenario. Reference regret is selected reference PRS minus best reference
PRS. Ties average across all minimizing candidates, avoiding arbitrary tie bias.
This reference objective is an oracle within the simulation, not empirical harm.
The independent model can assign zero inferred leakage to all candidates that
omit sensitive attributes, giving a broad tie. Recommendations use deterministic
score/attribute-name ordering to make output reproducible; tied scores are still
equivalent under PRS. Candidate rows use a fixed attribute order.

## Reproduction and paper use

Run `python src/run_experiments.py`; no additional dependencies beyond the
existing numpy and matplotlib requirements are required. Run analytic tests with
`python -m unittest discover -s tests -v`.

`metadata.json` stores config, versions, platform, source checksums, base commit,
semantics and limitations. Timing depends on hardware and cache reuse, so should
not be presented as a controlled scalability benchmark. Generated files are
ignored by git; rerun the experiment to obtain them. The generated experimental
section contains computed values and declared limitations, not manually invented
results. Its table uses the first seed and largest configured training set.
