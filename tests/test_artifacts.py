"""Check the published data contract and the real Streamlit display without retraining."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = json.loads((ROOT / 'data/model_evaluation.json').read_text())
        cls.leads = pd.read_csv(ROOT / 'data/top_leads.csv')
        cls.raw = pd.read_csv(ROOT / 'data/bank-full.csv', sep=';')

    def test_export_is_holdout_only_with_correct_outcomes_and_order(self):
        y = self.raw.y.eq('yes').astype(int)
        train, test = train_test_split(self.raw.index, test_size=self.e['split']['test_fraction'],
                                       stratify=y, random_state=self.e['split']['seed'])
        t = self.leads
        self.assertEqual(len(t), 100)
        self.assertTrue(t.source_row_id.is_unique)
        self.assertTrue(set(t.source_row_id).issubset(test))
        self.assertTrue(set(t.source_row_id).isdisjoint(train))
        self.assertEqual(t.actual_outcome.tolist(), y.loc[t.source_row_id].tolist())
        self.assertTrue(t.model_score.between(0, 1).all())
        ordered = t.sort_values(['model_score', 'source_row_id'], ascending=[False, True])
        self.assertEqual(t.source_row_id.tolist(), ordered.source_row_id.tolist())
        self.assertEqual(set(t.priority), {'Top 100'})
        self.assertEqual(self.e['holdout']['size'], len(test))
        self.assertEqual(self.e['holdout']['positive_count'], int(y.loc[test].sum()))

    def test_metrics_and_selected_model_have_consistent_provenance(self):
        e = self.e
        for path, key in [('bank-full.csv', 'source_sha256'), ('top_leads.csv', 'top_leads_sha256')]:
            self.assertEqual(hashlib.sha256((ROOT / 'data' / path).read_bytes()).hexdigest(), e[key])
        h, r = e['holdout'], e['ranking']
        p = int(self.leads.actual_outcome.sum())
        self.assertEqual(r['positives_captured_at_k'], p)
        self.assertAlmostEqual(h['positive_prevalence'], h['positive_count'] / h['size'])
        self.assertAlmostEqual(r['precision_at_k'], p / len(self.leads))
        self.assertAlmostEqual(r['capture_at_k'], p / h['positive_count'])
        self.assertAlmostEqual(r['lift_at_k'], r['precision_at_k'] / h['positive_prevalence'])
        rows = e['selection']['results']
        for row in rows:
            self.assertAlmostEqual(row['mean_average_precision'], np.mean(row['fold_average_precision']))
        winner = sorted(rows, key=lambda row: (-row['mean_average_precision'], row['model']))[0]
        self.assertEqual(e['selected_model'], winner['model'])
        self.assertTrue(set(e['features']).isdisjoint(e['excluded_features'] + ['y', 'source_row_id']))

    def test_notebook_is_executed_and_warning_free(self):
        nb = json.loads((ROOT / 'notebooks/01_eda.ipynb').read_text())
        cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertEqual([c['execution_count'] for c in cells], list(range(1, len(cells) + 1)))
        images = 0
        for cell in cells:
            for output in cell['outputs']:
                self.assertNotEqual(output['output_type'], 'error')
                self.assertNotIn('Warning:', ''.join(output.get('text', [])))
                images += 'image/png' in output.get('data', {})
        self.assertEqual(images, 2)

    def test_app_uses_generated_metrics_and_all_leads(self):
        app = AppTest.from_file(str(ROOT / 'app/app.py')).run(timeout=20)
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))
        actual = {item.label: item.value for item in app.metric}
        h, r = self.e['holdout'], self.e['ranking']
        expected = {
            'Holdout records': f"{h['size']:,}",
            'Holdout positive prevalence': f"{h['positive_prevalence']:.2%}",
            'Average precision': f"{h['average_precision']:.3f}",
            'ROC-AUC': f"{h['roc_auc']:.3f}",
            'Historical positives captured': f"{r['positives_captured_at_k']} / {r['top_k']}",
            'Precision@100': f"{r['precision_at_k']:.1%}",
            'Capture@100': f"{r['capture_at_k']:.2%}",
            'Lift@100': f"{r['lift_at_k']:.2f}×",
        }
        self.assertEqual(actual, expected)
        pd.testing.assert_frame_equal(app.dataframe[0].value, self.leads)


if __name__ == '__main__':
    unittest.main()
