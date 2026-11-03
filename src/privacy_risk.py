"""Exact discrete-BN leakage and ranking for small experiment models.

History denotes attribute variables, not observed holder values. Conditional
mutual information therefore averages over possible history values.
"""
import itertools
import math
from functools import lru_cache

import numpy as np


class ExactBN:
    """Exact enumeration over ancestors of queried variables (bounded in size)."""

    def __init__(self, model, attributes, max_states=2**22):
        self.order = tuple(model['order'])
        self.parents = model['parents']
        self.domains = {n: tuple(attributes[n]['states']) for n in self.order}
        self.max_states = max_states
        if len(set(self.order)) != len(self.order):
            raise ValueError('Duplicate variables in BN order')
        seen = set()
        self.tables = {}
        for name in self.order:
            ps = self.parents[name]
            if len(set(ps)) != len(ps) or not set(ps) <= seen:
                raise ValueError('Parents must precede each node in topological order')
            shape = [len(self.domains[p]) for p in ps] + [len(self.domains[name])]
            table = np.full(shape, np.nan)
            for row in model['cpts'][name]:
                key = tuple(self.domains[p].index(row['given'][p]) for p in ps)
                if np.isfinite(table[key]).all():
                    raise ValueError('Duplicate CPT parent configuration')
                table[key] = row['probabilities']
            if not np.isfinite(table).all() or (table < 0).any() or not np.allclose(table.sum(axis=-1), 1):
                raise ValueError('Invalid or incomplete CPT')
            self.tables[name] = table
            seen.add(name)

    def canonical(self, names):
        names = set(names)
        if not names <= set(self.order):
            raise ValueError(f'Unknown variables: {names - set(self.order)}')
        return tuple(n for n in self.order if n in names)

    def entropy(self, names):
        return self._entropy(self.canonical(names))

    @lru_cache(maxsize=1024)
    def _entropy(self, query):
        if not query:
            return 0.0
        ancestors = set(query)
        for n in reversed(self.order):
            if n in ancestors:
                ancestors.update(self.parents[n])
        order = [n for n in self.order if n in ancestors]
        sizes = [len(self.domains[n]) for n in order]
        count = math.prod(sizes)
        if count > self.max_states:
            raise ValueError(f'Exact inference needs {count} states; limit is {self.max_states}')
        flat = np.arange(count, dtype=np.int64)
        states = {}
        stride = count
        for n, size in zip(order, sizes):
            stride //= size
            states[n] = ((flat // stride) % size).astype(np.int16)
        probability = np.ones(count)
        for n in order:
            key = tuple(states[p] for p in self.parents[n]) + (states[n],)
            probability *= self.tables[n][key]
        keys = np.zeros(count, dtype=np.int64)
        for n in query:
            keys = keys * len(self.domains[n]) + states[n]
        marginal = np.bincount(keys, weights=probability,
                               minlength=math.prod(len(self.domains[n]) for n in query))
        if not np.isclose(marginal.sum(), 1.0):
            raise ValueError('Joint probability mass is not one')
        positive = marginal[marginal > 0]
        return float(-np.sum(positive * np.log2(positive)))

    def leakage(self, disclosure, target, history=()):
        """Return (normalized incremental leakage, residual entropy in bits).

        If residual entropy is numerically zero, additional leakage is defined
        as zero and saturation is reported. This convention does NOT imply that
        a previously exposed target remains private.
        """
        disclosure = set(self.canonical(disclosure))
        history = set(self.canonical(history))
        self.canonical([target])
        before = self.entropy(history | {target}) - self.entropy(history)
        if before <= 1e-12:
            return 0.0, 0.0
        after = self.entropy(history | disclosure | {target}) - self.entropy(history | disclosure)
        leakage = (before - after) / before
        if leakage < -1e-8 or leakage > 1 + 1e-8:
            raise ArithmeticError('Leakage outside information-theoretic bounds')
        return float(np.clip(leakage, 0, 1)), float(before)


def impact_weights(impacts, sensitive):
    """Normalize comparable impact magnitudes; never impute missing evidence."""
    sensitive = list(sensitive)
    if not sensitive or len(set(sensitive)) != len(sensitive):
        raise ValueError('Sensitive attributes must be nonempty and unique')
    values = np.array([impacts[s] for s in sensitive], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError('Impacts must be finite, nonnegative, and have positive total')
    return dict(zip(sensitive, (values / values.sum()).tolist()))


def enumerate_disclosures(policy, credentials, holder_values=None, minimal=False):
    """Enumerate all sets satisfying an OR of AND clauses.

    This is a small exhaustive experimental enumerator, NOT a PFA/SMT
    implementation. Clause predicates check holder eligibility when supplied.
    """
    available = set().union(*(set(c) for c in credentials)) if credentials else set()
    clauses = policy['any_of']
    universe = sorted(set().union(*(set(c) for c in clauses)))
    eligible = [c for c in clauses if set(c) <= available and
                (holder_values is None or all(holder_values.get(n) == v for n, v in c.items()))]
    feasible = []
    for size in range(len(universe) + 1):
        for candidate in itertools.combinations(universe, size):
            selected = set(candidate)
            if selected <= available and any(set(c) <= selected for c in eligible):
                if not minimal or not any(set(other) <= selected for other in feasible):
                    feasible.append(candidate)
    return feasible


def score_disclosures(engine, disclosures, sensitive, weights, history=()):
    weights = impact_weights(weights, sensitive)
    rows = []
    for i, disclosure in enumerate(disclosures, 1):
        leak, residual, saturated = {}, {}, []
        for target in sensitive:
            leak[target], residual[target] = engine.leakage(disclosure, target, history)
            if residual[target] == 0:
                saturated.append(target)
        contributions = {s: weights[s] * leak[s] for s in sensitive}
        rows.append({'id': f'D{i}', 'attributes': list(disclosure), 'history': list(history),
                     'leakage': leak, 'residual_entropy_bits': residual,
                     'saturated_targets': saturated, 'contributions': contributions,
                     'prs': sum(contributions.values())})
    return rows


def rank_and_filter(rows, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must lie in [0,1]')
    return sorted((r for r in rows if r['prs'] <= threshold),
                  key=lambda r: (r['prs'], tuple(r['attributes'])))
