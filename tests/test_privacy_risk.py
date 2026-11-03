"""Analytic distributions test exact inference, conditioning and selection."""
import sys
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from privacy_risk import ExactBN, impact_weights, enumerate_disclosures, score_disclosures, rank_and_filter


def binary_model(order, parents, probabilities):
    import itertools
    attributes = {n: {'states': ['0', '1']} for n in order}
    cpts = {}
    for n in order:
        cpts[n] = [{'given': dict(zip(parents[n], map(str, bits))), 'probabilities': [1 - p, p]}
                   for bits, p in zip(itertools.product([0, 1], repeat=len(parents[n])), probabilities[n])]
    return ExactBN({'order': order, 'parents': parents, 'cpts': cpts}, attributes)


class PrivacyRiskTests(unittest.TestCase):
    def test_independence(self):
        bn = binary_model(['A', 'T'], {'A': [], 'T': []}, {'A': [.5], 'T': [.3]})
        self.assertAlmostEqual(bn.leakage(['A'], 'T')[0], 0)

    def test_direct_and_repeated_disclosure(self):
        bn = binary_model(['T'], {'T': []}, {'T': [.5]})
        self.assertAlmostEqual(bn.leakage(['T'], 'T')[0], 1)
        self.assertEqual(bn.leakage(['T'], 'T', ['T']), (0, 0))

    def test_xor_joint_leakage_and_conditional_synergy(self):
        bn = binary_model(['A', 'B', 'T'], {'A': [], 'B': [], 'T': ['A', 'B']},
                          {'A': [.5], 'B': [.5], 'T': [0, 1, 1, 0]})
        self.assertAlmostEqual(bn.leakage(['A'], 'T')[0], 0)
        self.assertAlmostEqual(bn.leakage(['B'], 'T')[0], 0)
        self.assertAlmostEqual(bn.leakage(['A', 'B'], 'T')[0], 1)
        self.assertAlmostEqual(bn.leakage(['B'], 'T', ['A'])[0], 1)
        self.assertAlmostEqual(bn.leakage(['A'], 'T', ['A'])[0], 0)

    def test_d_separation_chain(self):
        bn = binary_model(['A', 'B', 'T'], {'A': [], 'B': ['A'], 'T': ['B']},
                          {'A': [.5], 'B': [.1, .9], 'T': [.2, .8]})
        self.assertGreater(bn.leakage(['A'], 'T')[0], 0)
        self.assertAlmostEqual(bn.leakage(['A'], 'T', ['B'])[0], 0)

    def test_weights_and_threshold_boundaries(self):
        self.assertEqual(impact_weights({'T': 5}, ['T']), {'T': 1})
        for values in [{'T': 0}, {'T': -1}, {'T': float('nan')}]:
            with self.assertRaises(ValueError):
                impact_weights(values, ['T'])
        rows = [{'prs': .1, 'attributes': ['A']}, {'prs': .2, 'attributes': ['B']}]
        self.assertEqual(len(rank_and_filter(rows, .1)), 1)
        self.assertEqual(rank_and_filter(rows, 0), [])
        with self.assertRaises(ValueError):
            rank_and_filter(rows, 1.1)

    def test_eligibility_credentials_and_supersets(self):
        policy = {'any_of': [{'A': 'yes', 'B': 'yes'}, {'C': 'yes'}]}
        credentials = [['A', 'B', 'C']]
        all_sets = enumerate_disclosures(policy, credentials, {'A': 'yes', 'B': 'yes', 'C': 'yes'})
        self.assertEqual(len(all_sets), 5)
        minimal = enumerate_disclosures(policy, credentials, {'A': 'yes', 'B': 'yes', 'C': 'no'}, True)
        self.assertEqual(minimal, [('A', 'B')])
        self.assertEqual(enumerate_disclosures(policy, [['A']], {'A': 'yes'}), [])

    def test_prs_matrix_identity(self):
        bn = binary_model(['A', 'T'], {'A': [], 'T': ['A']}, {'A': [.5], 'T': [.1, .9]})
        rows = score_disclosures(bn, [[], ['A'], ['T']], ['A', 'T'], {'A': 1, 'T': 3})
        matrix = np.array([[r['leakage'][s] for s in ['A', 'T']] for r in rows])
        np.testing.assert_allclose(matrix @ [.25, .75], [r['prs'] for r in rows])
        self.assertEqual(rows[0]['prs'], 0)

    def test_multistate_root(self):
        model = {'order': ['T'], 'parents': {'T': []},
                 'cpts': {'T': [{'given': {}, 'probabilities': [.25, .25, .5]}]}}
        bn = ExactBN(model, {'T': {'states': ['low', 'mid', 'high']}})
        self.assertAlmostEqual(bn.entropy(['T']), 1.5)
        self.assertAlmostEqual(bn.leakage(['T'], 'T')[0], 1)

class RunnerTests(unittest.TestCase):
    def test_experiment_artifacts_and_holdout(self):
        import csv
        import json
        import tempfile
        from run_experiments import run
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'config/experiments.json').read_text())
        config.update(seeds=[7], train_sizes=[40], test_size=20)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            metadata = run(config, output)
            self.assertEqual(metadata['candidate_count'], 37)
            self.assertEqual(metadata['minimal_candidate_count'], 3)
            with (output / 'scores.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 37 * 3 * 3)
            self.assertTrue(all(0 <= float(r['prs']) <= 1 for r in rows))
            with (output / 'thresholds.csv').open() as stream:
                threshold_rows = list(csv.DictReader(stream))
            for row in threshold_rows:
                if row['recommended_prs']:
                    self.assertLessEqual(float(row['recommended_prs']), float(row['threshold']))
            self.assertTrue((output / 'experimental_section.tex').exists())
            self.assertTrue((output / 'history_weights.pdf').exists())


if __name__ == '__main__':
    unittest.main()
