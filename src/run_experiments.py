"""Reproducible evaluation artifacts for the RDSD paper; no real incident data."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from run_demo import SPEC, NAMES, generate, learn, nll
from privacy_risk import ExactBN, enumerate_disclosures, impact_weights, score_disclosures, rank_and_filter

ROOT = Path(__file__).resolve().parents[1]


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_model(train):
    result = {'order': NAMES, 'parents': {n: [] for n in NAMES}, 'cpts': {}}
    for j, name in enumerate(NAMES):
        counts = np.bincount(train[:, j], minlength=2)
        result['cpts'][name] = [{'given': {}, 'probabilities': ((counts + 1) / (len(train) + 2)).tolist()}]
    return result


def run(config, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'models').mkdir(exist_ok=True)
    attrs = {name: {'states': states} for name, states, _, _ in SPEC}
    candidates = enumerate_disclosures(config['policy'], config['credentials'], config['holder_values'])
    minimal = enumerate_disclosures(config['policy'], config['credentials'], config['holder_values'], minimal=True)
    if not candidates:
        raise ValueError('Experiment requires at least one eligible disclosure')
    sensitive = config['sensitive']
    if set(sensitive) - set(NAMES):
        raise ValueError('Unknown sensitive attributes')
    sizes = config['train_sizes']
    if not sizes or min(sizes) < 1 or config['test_size'] < 1:
        raise ValueError('Positive train/test sizes are required')
    if not config['seeds']:
        raise ValueError('At least one replicate seed is required')
    all_scores, comparisons, thresholds, validation, timings, accuracy = [], [], [], [], [], []
    first_examples = None
    start_all = time.perf_counter()
    for seed in config['seeds']:
        profiles, reference = generate(max(sizes) + config['test_size'], seed)
        permutation = np.random.default_rng(seed + 10000).permutation(len(profiles))
        pool, test = profiles[permutation[:max(sizes)]], profiles[permutation[max(sizes):]]
        reference_engine = ExactBN(reference, attrs)
        for size in sizes:
            train = pool[:size]
            start = time.perf_counter()
            model = learn(train, config['max_parents'])
            training_seconds = time.perf_counter() - start
            engine = ExactBN(model, attrs)
            independent = independent_model(train)
            independent_engine = ExactBN(independent, attrs)
            truth = {(p, n) for n in NAMES for p in reference['parents'][n]}
            edges = {(p, n) for n in NAMES for p in model['parents'][n]}
            tp = len(truth & edges)
            validation.append({'seed': seed, 'train_size': size, 'test_size': len(test),
                               'bn_test_nll_nats': nll(test, model),
                               'independent_test_nll_nats': nll(test, independent),
                               'reference_test_nll_nats': nll(test, reference),
                               'edge_precision': tp / len(edges) if edges else 0,
                               'edge_recall': tp / len(truth), 'learned_edges': len(edges),
                               'training_seconds': training_seconds})
            (output / 'models' / f'bn_seed{seed}_n{size}.json').write_text(json.dumps(model, indent=2))
            for history_name, history in config['histories'].items():
                # First pass per history populates exact-entropy caches; time it explicitly.
                start = time.perf_counter()
                raw = score_disclosures(engine, candidates, sensitive, {s: 1 for s in sensitive}, history)
                score_seconds = time.perf_counter() - start
                timings.append({'seed': seed, 'train_size': size, 'history': history_name,
                                'candidate_count': len(candidates), 'scoring_seconds': score_seconds,
                                'cache_policy': 'same model cache reused across histories; first weight scenario only'})
                ref = score_disclosures(reference_engine, candidates, sensitive, {s: 1 for s in sensitive}, history)
                ind = score_disclosures(independent_engine, candidates, sensitive, {s: 1 for s in sensitive}, history)
                accuracy.append({'seed': seed, 'train_size': size, 'history': history_name,
                                 'leakage_mae': float(np.mean([abs(a['leakage'][s] - b['leakage'][s]) for a, b in zip(raw, ref) for s in sensitive]))})
                for scenario, impacts in config['impact_scenarios'].items():
                    weights = impact_weights(impacts, sensitive)
                    def reweight(rows):
                        return [dict(r, prs=sum(weights[s] * r['leakage'][s] for s in sensitive),
                                     contributions={s: weights[s] * r['leakage'][s] for s in sensitive}) for r in rows]
                    rows, ref_rows, ind_rows = reweight(raw), reweight(ref), reweight(ind)
                    if first_examples is None:
                        first_examples = {'rows': rows, 'weights': weights, 'seed': seed,
                                          'train_size': size, 'history': history_name, 'scenario': scenario}
                    if seed == config['seeds'][0] and size == max(sizes) and history_name == 'none' and scenario == 'equal':
                        first_examples = {'rows': rows, 'weights': weights, 'seed': seed,
                                          'train_size': size, 'history': history_name, 'scenario': scenario}
                    for row in rows:
                        flat = {'seed': seed, 'train_size': size, 'history': history_name,
                                'impact_scenario': scenario, 'disclosure': '+'.join(row['attributes']),
                                'attribute_count': len(row['attributes']), 'prs': row['prs'],
                                'saturated_targets': '+'.join(row['saturated_targets'])}
                        flat.update({f'leakage_{s}': row['leakage'][s] for s in sensitive})
                        flat.update({f'weight_{s}': weights[s] for s in sensitive})
                        all_scores.append(flat)
                    true_scores = np.array([r['prs'] for r in ref_rows])
                    selected_metrics = {
                        'rdsd': np.array([r['prs'] for r in rows]),
                        'independent_bn': np.array([r['prs'] for r in ind_rows]),
                        'minimum_cardinality': np.array([len(r['attributes']) for r in rows], dtype=float),
                        'additive_attribute_cost': np.array([sum(config['attribute_costs'][a] for a in r['attributes']) for r in rows])}
                    for baseline, metric in selected_metrics.items():
                        tied = np.flatnonzero(np.isclose(metric, metric.min(), atol=1e-12, rtol=0))
                        chosen_risk = float(true_scores[tied].mean())
                        comparisons.append({'seed': seed, 'train_size': size, 'history': history_name,
                                            'impact_scenario': scenario, 'method': baseline,
                                            'tied_minima': len(tied), 'reference_prs_selected': chosen_risk,
                                            'reference_regret': chosen_risk - float(true_scores.min()),
                                            'reference_optimal_fraction': float(np.isclose(true_scores[tied], true_scores.min(), atol=1e-10).mean())})
                    comparisons.append({'seed': seed, 'train_size': size, 'history': history_name,
                                        'impact_scenario': scenario, 'method': 'uniform_random',
                                        'tied_minima': len(rows), 'reference_prs_selected': float(true_scores.mean()),
                                        'reference_regret': float(true_scores.mean() - true_scores.min()),
                                        'reference_optimal_fraction': float(np.isclose(true_scores, true_scores.min(), atol=1e-10).mean())})
                    for tau in config['thresholds']:
                        ranked = rank_and_filter(rows, tau)
                        thresholds.append({'seed': seed, 'train_size': size, 'history': history_name,
                                           'impact_scenario': scenario, 'threshold': tau,
                                           'accepted_count': len(ranked), 'candidate_count': len(rows),
                                           'recommended': '+'.join(ranked[0]['attributes']) if ranked else '',
                                           'recommended_prs': ranked[0]['prs'] if ranked else ''})
            print(f'Completed seed={seed}, train_size={size}', flush=True)
    csv_write(output / 'scores.csv', all_scores)
    csv_write(output / 'baselines.csv', comparisons)
    csv_write(output / 'thresholds.csv', thresholds)
    csv_write(output / 'validation.csv', validation)
    csv_write(output / 'timings.csv', timings)
    csv_write(output / 'leakage_accuracy.csv', accuracy)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = 'unavailable'
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [ROOT / 'src' / n for n in ['run_demo.py', 'privacy_risk.py', 'run_experiments.py']]}
    metadata = {'config': config, 'python': platform.python_version(), 'numpy': np.__version__,
                'platform': platform.platform(), 'base_commit': commit, 'source_sha256': source_hashes,
                'core_experiment_seconds': time.perf_counter() - start_all,
                'candidate_count': len(candidates), 'minimal_candidate_count': len(minimal),
                'data_kind': 'synthetic', 'candidate_generation': 'exhaustive OR-of-AND experimental enumerator; not PFA/SMT',
                'history_semantics': 'conditional mutual information averaged over history values; not observed-value conditioning',
                'zero_entropy_policy': 'zero additional leakage with saturation flag; impacts remain normalized over original sensitive set',
                'limitations': ['No ITAP data or empirical impact estimation', 'No wallet integration or VP construction',
                                'Supplied variable ordering and at most configured parents',
                                'Exact enumeration bounded by ancestor state-space size',
                                'Only one reference dependency scenario with repeated samples',
                                'Baselines are transparent experimental comparators, not a reproduction of Saha et al.',
                                'Reported conditional leakage is incremental, not cumulative total leakage']}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    make_figures(output, first_examples, all_scores, validation, config)
    write_scenario_summary(output, all_scores, config, minimal)
    write_paper(output, first_examples, comparisons, validation, config, metadata, all_scores, thresholds)
    return metadata


def write_scenario_summary(output, scores, config, minimal):
    minimal_sets = {frozenset(d) for d in minimal}
    summary = []
    for history in config['histories']:
        for scenario in config['impact_scenarios']:
            rows = [r for r in scores if r['seed'] == config['seeds'][0] and
                    r['train_size'] == max(config['train_sizes']) and
                    r['history'] == history and r['impact_scenario'] == scenario and
                    frozenset(r['disclosure'].split('+')) in minimal_sets]
            for row in sorted(rows, key=lambda r: r['prs']):
                summary.append({k: row[k] for k in ['seed', 'train_size', 'history', 'impact_scenario', 'disclosure', 'prs']})
    csv_write(output / 'scenario_summary.csv', summary)


def make_figures(output, example, scores, validation, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    minimal_sets = {frozenset(d) for d in enumerate_disclosures(config['policy'], config['credentials'], config['holder_values'], True)}
    rows = [r for r in example['rows'] if frozenset(r['attributes']) in minimal_sets]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(rows))
    for i, s in enumerate(config['sensitive']):
        ax.bar(x + (i - 1) * .23, [r['leakage'][s] for r in rows], .23, label=s)
    ax.set_xticks(x, [' +\n'.join(r['attributes']) for r in rows], fontsize=8)
    ax.set_ylabel('Normalized inference leakage')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'leakage.pdf'); fig.savefig(output / 'leakage.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for key, label in [('bn_test_nll_nats', 'Learned BN'), ('independent_test_nll_nats', 'Independent baseline')]:
        means, errors = [], []
        for size in config['train_sizes']:
            values = [r[key] for r in validation if r['train_size'] == size]
            means.append(np.mean(values)); errors.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        ax.errorbar(config['train_sizes'], means, yerr=errors, marker='o', capsize=3, label=label)
    ax.set_xlabel('Training profiles'); ax.set_ylabel('Held-out NLL (nats per profile)')
    ax.legend(); fig.tight_layout()
    fig.savefig(output / 'sample_size.pdf'); fig.savefig(output / 'sample_size.png', dpi=180); plt.close(fig)


    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    labels = [' + '.join(sorted(d)) for d in sorted(minimal_sets, key=lambda x: sorted(x))]
    ordered_sets = sorted(minimal_sets, key=lambda x: sorted(x))
    for ax, group_key, fixed_key, fixed_value, title in [
        (axes[0], 'history', 'impact_scenario', 'equal', 'Prior disclosures; equal weights'),
        (axes[1], 'impact_scenario', 'history', 'none', 'Impact scenarios; no prior disclosure')]:
        groups = list(config['histories'] if group_key == 'history' else config['impact_scenarios'])
        for i, group in enumerate(groups):
            selected = [r for r in scores if r['seed'] == example['seed'] and r['train_size'] == max(config['train_sizes']) and r[group_key] == group and r[fixed_key] == fixed_value]
            values = [next(r['prs'] for r in selected if frozenset(r['disclosure'].split('+')) == d) for d in ordered_sets]
            ax.plot(range(len(values)), values, marker='o', label=group.replace('_', ' '))
        ax.set_xticks(range(len(labels)), [label.replace(' + ', '\n+ ') for label in labels], fontsize=7)
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7); ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Incremental PRS')
    fig.tight_layout()
    fig.savefig(output / 'history_weights.pdf'); fig.savefig(output / 'history_weights.png', dpi=180); plt.close(fig)


def write_paper(output, example, comparisons, validation, config, metadata, scores, thresholds):
    minimal_sets = {frozenset(d) for d in enumerate_disclosures(config['policy'], config['credentials'], config['holder_values'], True)}
    rows = [r for r in example['rows'] if frozenset(r['attributes']) in minimal_sets]
    def esc(text):
        return str(text).replace('_', r'\_').replace('&', r'\&')
    table = ['\\begin{table}[htbp]', '\\centering', '\\caption{Inference leakage and PRS for inclusion-minimal candidates.}',
             '\\label{tab:experiment-leakage}', '\\small', '\\begin{tabular}{lrrrr}', '\\hline',
             'Disclosure & Income & Debt & Health & PRS \\\\', '\\hline']
    for row in rows:
        table.append(' + '.join(row['attributes']) + ' & ' + ' & '.join(f"{row['leakage'][s]:.4f}" for s in config['sensitive']) + f" & {row['prs']:.4f} " + r'\\')
    table += ['\\hline', '\\end{tabular}', '\\end{table}']
    (output / 'results_table.tex').write_text('\n'.join(table) + '\n')
    selected = [r for r in validation if r['train_size'] == max(config['train_sizes'])]
    mean_bn = float(np.mean([r['bn_test_nll_nats'] for r in selected]))
    mean_ind = float(np.mean([r['independent_test_nll_nats'] for r in selected]))
    baseline = [r for r in comparisons if r['train_size'] == max(config['train_sizes']) and r['history'] == 'none' and r['impact_scenario'] == 'equal']
    lines = [r'\section{Experimental Evaluation}', r'\label{sec:experiments}',
             r'\subsection{Setup}',
             f"We evaluate a synthetic reference Bayesian network with {len(NAMES)} binary attributes, using {len(config['seeds'])} generation seeds. "
             f"Training sizes are {', '.join(map(str, config['train_sizes']))}, with {config['test_size']} disjoint held-out profiles per seed. "
             'Training subsets are nested within each seed. Structure learning uses ordered BIC search, Laplace-smoothed CPTs, '
             f"and at most {config['max_parents']} parents per node. The attribute order is supplied.",
             f"The experimental policy has three alternative eligibility clauses. Exhaustive enumeration over its six attributes produces {metadata['candidate_count']} compliant disclosure sets, including {metadata['minimal_candidate_count']} inclusion-minimal sets. "
             'A configured hypothetical holder satisfies all three clauses. This enumerator supports the experiment and is not a PFA/SMT implementation.',
             r'\subsection{Evaluation Questions}',
             'We assess whether equally sized disclosures have different leakage, whether dependency-aware ranking improves selection under the known reference distribution, '
             'how rankings change with prior disclosures and impact weights, how the risk threshold filters candidates, and how model quality changes with training size.',
             'Equal, income-priority, and health-priority impact scenarios are hypothetical. No ITAP data or empirical impact estimation is used. '
             'History scenarios contain no known attributes, AgeBand alone, or AgeBand and Region. Conditional mutual information averages over history values. '
             'If residual target entropy is zero, additional leakage is set to zero and the target is flagged as already determined; this does not mean that the target remains private.',
             r'\subsection{Results}',
             f"At the largest training size, the mean held-out negative log likelihood is {mean_bn:.4f} nats per profile for the learned BN and {mean_ind:.4f} for the independent baseline. "
             'The CSV artifacts retain individual-seed results and execution times. Error bars in the sample-size figure represent sample standard deviations, not confidence intervals.',
             f"Table~\\ref{{tab:experiment-leakage}} reports the first configured seed at training size {example['train_size']}, with equal weights and no prior disclosure.",
             r'\input{results_table.tex}',
             'Baseline selection is evaluated using PRS under the known synthetic reference network. Regret is the selected reference PRS minus the lowest reference PRS among the same candidates. '
             'All tied minimizers are averaged rather than choosing an arbitrary favorable tie. The additive comparator uses configured public-attribute costs and does not reproduce the full Saha et al. framework.']
    for method in sorted({r['method'] for r in baseline}):
        vals = [r['reference_regret'] for r in baseline if r['method'] == method]
        lines.append(f"Mean reference regret for {esc(method)} is {np.mean(vals):.6f} across the configured seeds.")
    lines.append(r'\paragraph{Impact sensitivity and prior disclosures.}')
    for scenario in config['impact_scenarios']:
        candidates = [r for r in scores if r['seed'] == example['seed'] and r['train_size'] == max(config['train_sizes']) and r['history'] == 'none' and r['impact_scenario'] == scenario]
        best = min(candidates, key=lambda r: (r['prs'], r['disclosure']))
        lines.append(f"In the {esc(scenario)} scenario, a minimum-score disclosure is {esc(best['disclosure'].replace('+', ' and '))}, with PRS {best['prs']:.4f} (first seed, largest training set).")
    lines.append('When AgeBand and Region are already known, repeating that same pair contributes zero additional information. This is a property of incremental leakage, not evidence that prior exposure is harmless.')
    lines.append(r'\paragraph{Threshold filtering.}')
    for row in thresholds:
        if row['seed'] == example['seed'] and row['train_size'] == max(config['train_sizes']) and row['history'] == 'none' and row['impact_scenario'] == 'equal' and row['threshold'] in [.05, .075, .1]:
            lines.append(f"At threshold {row['threshold']:.3f}, {row['accepted_count']} of {row['candidate_count']} candidates are retained (first seed, equal weights, no history).")
    lines += [r'\subsection{Limitations}',
              'The evaluation demonstrates the scoring workflow under a single synthetic dependency scenario. It does not validate real-world harm estimates or deployed wallet performance. '
              'Exact inference scales exponentially with the ancestor state space. Prior-disclosure experiments measure incremental population-average leakage, not observed-value or worst-case leakage. '
              'Learning from heterogeneous population profiles is not equivalent to learning a representative dependency model from one holder\'s presentation history. '
              'Normalized impact weights require comparable consequence units; normalization alone does not resolve heterogeneous units.']
    (output / 'experimental_section.tex').write_text('\n\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/experiments.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/experiments')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    metadata = run(config, args.output)
    print(json.dumps({'output': str(args.output.resolve()), 'seconds': metadata['core_experiment_seconds'],
                      'candidates': metadata['candidate_count']}, indent=2))


if __name__ == '__main__':
    main()
