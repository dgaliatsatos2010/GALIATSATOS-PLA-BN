# GALIATSATOS method specification

Status: manuscript-frozen research specification, version 1.0.0 (21 August 2026).

GALIATSATOS is the proposed name of a supervised machine-learning method for
aligning outcomes recorded under heterogeneous, documented definitions. The
name is an eponym, not an acronym. PLA-BN is the first reference estimator.

## Inputs and outputs

Training input:

- environment feature matrices `X_e` with the same predictor columns;
- observed label vectors `y_e`;
- one known column-stochastic operator `C_e` per environment, where
  `C_e[l, z] = P(Y_e=l | Z=z, e)`;
- the requested canonical class count `K`.

Fitted outputs:

- definition-identifiability diagnostics;
- a canonical predictor `p(Z | X)`;
- `transform(X)`, the K-dimensional canonical posterior representation;
- canonical hard predictions;
- transported probabilities and predictions for a target operator `C_star`.

## Fixed six-stage algorithm

1. Declare the canonical finite latent outcome space `Z = {0, ..., K-1}` and
   document the meaning of every class. Ordered outcomes are an important
   special case, not a requirement of the estimator.
2. Encode every environment's observed definition as `C_e` and verify that its
   columns are nonnegative and sum to one.
3. Gate identifiability by stacking the operators vertically. Record the rank,
   singular values, and condition number. Stop by default if `rank(C) < K`.
4. Initialize the canonical prior from weighted environment label proportions
   and initialize latent responsibilities compatible with each observed label.
5. Learn the PLA-BN estimator by guarded multi-start Structural EM. Start 0
   uses the deterministic operator/prior initialization; later starts perturb
   only compatible latent responsibilities. Each start alternates the
   coarsening-aware E-step, responsibility-weighted parameter estimation, and
   TAN structure updates. Retain the start with the highest observed-data
   likelihood and record an explicit termination reason.
6. Transform a record to `g(x) = p_hat(Z | x)`. Predict canonically with
   `argmax(g(x))`, or transport to a target definition with
   `p_hat(Y_star | x) = C_star @ g(x)`.

## Pseudocode

```text
GALIATSATOS({X_e, y_e, C_e}, K, settings):
    validate all inputs and operators
    C <- vertical_stack(C_1, ..., C_E)
    diagnostics <- SVD_and_rank(C)
    if diagnostics.rank < K and require_identifiable:
        stop without fitting

    pi <- projected_weighted_least_squares({y_e, C_e})
    q0 <- compatible_responsibilities({y_e, C_e}, pi)

    for start in 1..n_init:
        q <- q0 for start 1, otherwise compatible_random_perturbation(q0)
        tree, theta <- initialize_PLA_BN({X_e}, q)
        repeat until tolerance, guarded stop, or max_iter:
            q <- posterior_responsibilities({X_e, y_e, C_e}, tree, theta)
            candidate_tree <- TAN_maximum_spanning_tree({X_e}, q)
            candidate_theta <- weighted_CPT_update({X_e}, q, candidate_tree)
            reject likelihood-decreasing structure/parameter updates
        save final observed-data likelihood and termination diagnostics

    retain the start with maximum final observed-data likelihood
    return fitted method, diagnostics, and likelihood history

transform(X):
    return p_hat(Z | X)

transport_proba(X, C_star):
    return transform(X) @ transpose(C_star)
```

## Partition-lattice interpretation

For deterministic definitions, each operator induces a partition `Pi_e` of the
canonical label space: two canonical states lie in the same block when the
observed definition maps them to the same label. The set of such partitions can
be ordered by refinement, giving the partition-lattice interpretation behind
the name PLA-BN. Multiple coarse partitions can therefore supply complementary
resolution. Stochastic column-stochastic operators are treated as a generalized
coarsening extension rather than literal set partitions.

The rank gate is an **operator-level or definition-level identifiability
condition**. Full column rank of the vertically stacked operators makes the
linear map from canonical class-probability vectors to the collection of
observed-definition probability vectors injective. It is not, by itself, a
proof of global statistical identifiability of every BN parameter under
arbitrary population shifts.

## Optimization diagnostics in v0.5.0

The fitted object reports `converged_`, `termination_reason_`, `n_iter_`,
`final_log_likelihood_`, `relative_last_improvement_`, `structure_changes_`,
`best_start_`, and per-start diagnostics. `termination_reason_` distinguishes
`tolerance`, `max_iter`, and `likelihood_guard`. A guarded stop is reported
explicitly rather than being silently labelled as convergence.

## Leakage-safe evaluation rule

The method uses training labels to learn its canonical representation. In every
cross-validation or bootstrap resample, instantiate and fit a fresh method using
only the training partition. Transform the validation or test partition only
after that fit. Never fit the method on the complete dataset before splitting.

For nested evaluation, the outer test labels must not be accepted by the fit
API. Any learned imputation, feature filtering, discretization, environment
construction, or estimator setting must be fitted inside the outer training
partition. Hyperparameters must be selected in inner folds; when canonical
labels are unavailable by design, selection must use an observed-data criterion
such as the coarsened-label log loss. Exact duplicate records must be grouped
so that they cannot cross a train/test boundary.

## Scientific status

This specification and implementation demonstrate a coherent named interface;
they do not prove absolute novelty, patentability, freedom to operate, adoption
of the name, or predictive superiority. Evidence now includes controlled
simulation and nested validation on three real benchmark datasets, but the
heterogeneous definitions in those benchmarks are imposed semi-synthetically
inside training folds and the proposed method still uses one estimator backend.
Natural multi-study validation and independent review are required before a
journal submission can make broader claims.
