import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visualizer import Visualizer


class PerformanceChartTests(unittest.TestCase):
    def test_chart_does_not_invent_an_unprovided_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            visualizer = Visualizer(pd.DataFrame(), graph_dir=directory)
            visualizer._save = Mock()
            results = dict(accuracy=0.6, precision=0.5, recall=0.4, f1=0.44, auc=0.7)
            visualizer.plot_model_comparison(results)
            figure = visualizer._save.call_args.args[0]
            self.assertEqual(len(figure.axes[0].patches), 5)
            self.assertEqual([bar.get_height() for bar in figure.axes[0].patches],
                             [60, 50, 40, 44, 70])
            self.assertFalse(any("+13" in text.get_text() for text in figure.texts))
            plt.close(figure)

    def test_explicit_baseline_is_used_without_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            visualizer = Visualizer(pd.DataFrame(), graph_dir=directory)
            visualizer._save = Mock()
            current = dict(accuracy=0.6, precision=0.5, recall=0.4, f1=0.44, auc=0.7)
            baseline = dict(accuracy=0.2, precision=0.1, recall=0.3, f1=0.15, auc=0.5)
            visualizer.plot_model_comparison(current, baseline)
            figure = visualizer._save.call_args.args[0]
            self.assertEqual([bar.get_height() for bar in figure.axes[0].patches[:5]],
                             [20, 10, 30, 15, 50])
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
