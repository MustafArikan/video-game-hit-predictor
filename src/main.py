"""
main.py
-------
Entry point for the Video Game Hit Predictor project.

Run this file from the project root directory:
    python src/main.py

Pipeline:
    Step 1 - DataLoader   : load CSV, clean, engineer features, encode
    Step 2 - DataAnalyzer : compute NumPy statistics, save text report
    Step 3 - Visualizer   : generate 7 data charts (PNG files)
    Step 4 - ModelTrainer : train DNN + XGBoost ensemble, evaluate
    Step 5 - Visualizer   : generate confusion matrix chart (chart 8)

All outputs are saved to the outputs/ folder.
"""

import os
import sys

# Add the src/ directory to the Python path so imports work
# regardless of which directory the user runs the script from
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import DataLoader
from analyzer    import DataAnalyzer
from visualizer  import Visualizer
from model_trainer import ModelTrainer


# ── File paths (relative to the project root) ─────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data",
                            "Video_Games_Sales_as_at_22_Dec_2016.csv")
GRAPH_DIR    = os.path.join(PROJECT_ROOT, "outputs", "graphs")
REPORT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "reports")


def print_banner(text: str):
    """Print a formatted section header to the console."""
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def main():
    print("\n" + "=" * 60)
    print("  VIDEO GAME HIT PREDICTOR")
    print("  DNN + XGBoost Ensemble | Target Encoding | SMOTE")
    print("=" * 60)

    # ── Step 1: Load and prepare data ─────────────────────────────────────────
    print_banner("STEP 1: Data Loading & Feature Engineering")

    loader = DataLoader(DATA_PATH)
    loader.load_and_prepare()

    # Get the human-readable dataframe for analysis and charts
    df_clean = loader.df_clean

    # Get the numeric arrays for the model
    X, y = loader.get_features_and_target()

    print(f"\n  Feature matrix : {X.shape}  ({X.shape[1]} features)")
    print(f"  Target vector  : {y.shape}")
    print(f"  Positive rate  : {y.mean():.2%} (hits in the full dataset)")

    # ── Step 2: Statistical analysis ──────────────────────────────────────────
    print_banner("STEP 2: Data Analysis with NumPy & Pandas")

    analyzer = DataAnalyzer(df_clean, report_dir=REPORT_DIR)
    analysis = analyzer.run_all()

    # Print a quick statistics summary to the console
    print("\n  Global Sales statistics (NumPy):")
    gs = analysis["statistics"]["Global_Sales"]
    for stat, value in gs.items():
        print(f"    {stat:<8}: {value:.3f} M copies")

    # ── Step 3: Data visualizations ───────────────────────────────────────────
    print_banner("STEP 3: Generating Data Charts (1–7)")

    visualizer = Visualizer(df_clean, graph_dir=GRAPH_DIR)
    visualizer.plot_all_data_charts()

    # ── Step 4: Model training and evaluation ─────────────────────────────────
    print_banner("STEP 4: Training DNN + XGBoost Ensemble")

    trainer = ModelTrainer(X, y, report_dir=REPORT_DIR)
    results = trainer.train()

    # ── Step 5: Confusion matrix chart ────────────────────────────────────────
    print_banner("STEP 5: Confusion Matrix Chart")

    cm = trainer.get_confusion_matrix()
    visualizer.plot_confusion_matrix(cm)

    # ── Step 6: Before vs after comparison chart ───────────────────────────────
    print_banner("STEP 6: Model Comparison Chart (Baseline vs Improved)")

    visualizer.plot_model_comparison(results)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Threshold : {trainer.threshold:.2f}  (optimized on val set)")
    print(f"  Accuracy  : {results['accuracy']*100:.2f}%")
    print(f"  Precision : {results['precision']*100:.2f}%")
    print(f"  Recall    : {results['recall']*100:.2f}%")
    print(f"  F1 Score  : {results['f1']*100:.2f}%")
    print(f"  AUC-ROC   : {results['auc']*100:.2f}%")
    print(f"\n  9 charts saved -> {GRAPH_DIR}")
    print(f"  Reports -> {REPORT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
