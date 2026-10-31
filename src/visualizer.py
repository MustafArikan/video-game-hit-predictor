"""
visualizer.py
-------------
Visualizer class:
  Produces 8 publication-quality Matplotlib charts and saves them
  to outputs/graphs/.

  1. Top-10 games by global sales (horizontal bar)
  2. Genre average sales (bar)
  3. Platform total sales (bar)
  4. Year-over-year sales trend (line)
  5. Regional sales comparison NA vs EU vs JP (grouped bar)
  6. Critic Score vs Global Sales (scatter + NumPy polyfit trend)
  7. Top-15 publisher average sales (bar)
  8. Model confusion matrix (heatmap) – called after model training
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works on servers / CI)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Consistent color palette
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63",
           "#9C27B0", "#00BCD4", "#FF5722", "#607D8B"]


class Visualizer:
    """Generates and saves all project charts."""

    def __init__(self, df: pd.DataFrame, graph_dir: str = "outputs/graphs"):
        # df = cleaned, human-readable dataframe (NOT encoded)
        self.df = df
        self.graph_dir = graph_dir
        os.makedirs(graph_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _save(self, fig, filename: str, comment: str = ""):
        """Save figure and print comment."""
        path = os.path.join(self.graph_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if comment:
            print(f"[Visualizer] {filename} → {comment}")

    # ------------------------------------------------------------------
    # Chart 1 – Top-10 games by global sales (horizontal bar)
    # ------------------------------------------------------------------

    def plot_top10_games(self):
        top10 = (
            self.df.nlargest(10, "Global_Sales")[["Name", "Global_Sales"]]
            .sort_values("Global_Sales")
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top10["Name"], top10["Global_Sales"], color=PALETTE[0])
        ax.bar_label(bars, fmt="%.1f M", padding=3, fontsize=9)
        ax.set_xlabel("Global Sales (million copies)")
        ax.set_title("Top 10 Video Games by Global Sales")
        ax.set_xlim(0, top10["Global_Sales"].max() * 1.15)
        self._save(fig, "01_top10_games.png",
                   "Wii Sports dominates; Nintendo titles lead the all-time list.")

    # ------------------------------------------------------------------
    # Chart 2 – Genre average sales (bar)
    # ------------------------------------------------------------------

    def plot_genre_sales(self):
        genre_avg = (
            self.df.groupby("Genre")["Global_Sales"]
            .mean()
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(genre_avg))]
        bars = ax.bar(genre_avg.index, genre_avg.values, color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_ylabel("Average Global Sales (million copies)")
        ax.set_title("Average Global Sales by Genre")
        ax.tick_params(axis="x", rotation=30)
        self._save(fig, "02_genre_avg_sales.png",
                   "Platform and Sports genres have the highest average sales per title.")

    # ------------------------------------------------------------------
    # Chart 3 – Platform total sales (bar)
    # ------------------------------------------------------------------

    def plot_platform_sales(self):
        plat = (
            self.df.groupby("Platform")["Global_Sales"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(plat.index, plat.values, color=PALETTE[1])
        ax.set_ylabel("Total Global Sales (million copies)")
        ax.set_title("Top 15 Platforms by Total Global Sales")
        ax.tick_params(axis="x", rotation=45)
        self._save(fig, "03_platform_sales.png",
                   "PS2 leads total sales; DS and Wii close behind.")

    # ------------------------------------------------------------------
    # Chart 4 – Year-over-year sales trend (line)
    # ------------------------------------------------------------------

    def plot_year_trend(self):
        df_year = (
            self.df[self.df["Year_of_Release"].between(2000, 2016)]
            .groupby("Year_of_Release")["Global_Sales"]
            .sum()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_year["Year_of_Release"], df_year["Global_Sales"],
                marker="o", color=PALETTE[2], linewidth=2.5)
        ax.fill_between(df_year["Year_of_Release"], df_year["Global_Sales"],
                        alpha=0.15, color=PALETTE[2])
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Global Sales (million copies)")
        ax.set_title("Annual Video Game Sales Trend (2000–2016)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        self._save(fig, "04_year_trend.png",
                   "Sales peaked around 2008–2009, declining after digital shift.")

    # ------------------------------------------------------------------
    # Chart 5 – Regional comparison NA vs EU vs JP (grouped bar)
    # ------------------------------------------------------------------

    def plot_regional_comparison(self):
        df_year = (
            self.df[self.df["Year_of_Release"].between(2000, 2016)]
            .groupby("Year_of_Release")[["NA_Sales", "EU_Sales", "JP_Sales"]]
            .sum()
            .reset_index()
        )
        years = df_year["Year_of_Release"].values
        x = np.arange(len(years))
        width = 0.28

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - width, df_year["NA_Sales"], width, label="North America", color=PALETTE[0])
        ax.bar(x,         df_year["EU_Sales"], width, label="Europe",        color=PALETTE[1])
        ax.bar(x + width, df_year["JP_Sales"], width, label="Japan",         color=PALETTE[3])

        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)
        ax.set_ylabel("Sales (million copies)")
        ax.set_title("Regional Sales Comparison by Year (NA vs EU vs JP)")
        ax.legend()
        self._save(fig, "05_regional_comparison.png",
                   "North America consistently leads; Japan shows steady but smaller volume.")

    # ------------------------------------------------------------------
    # Chart 6 – Critic Score vs Global Sales (scatter + polyfit trend)
    # ------------------------------------------------------------------

    def plot_critic_vs_sales(self):
        df_valid = self.df.dropna(subset=["Critic_Score", "Global_Sales"])
        # Cap outliers for readability
        df_valid = df_valid[df_valid["Global_Sales"] < 15]

        x = df_valid["Critic_Score"].values
        y = df_valid["Global_Sales"].values

        # NumPy polyfit – degree-1 (linear) trend line
        coeffs = np.polyfit(x, y, deg=1)
        trend_x = np.linspace(x.min(), x.max(), 200)
        trend_y = np.polyval(coeffs, trend_x)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(x, y, alpha=0.25, s=15, color=PALETTE[0], label="Games")
        ax.plot(trend_x, trend_y, color=PALETTE[3], linewidth=2.5,
                label=f"Trend  y={coeffs[0]:.3f}x+{coeffs[1]:.2f}")
        ax.set_xlabel("Critic Score (0–100)")
        ax.set_ylabel("Global Sales (million copies)")
        ax.set_title("Critic Score vs Global Sales")
        ax.legend()
        self._save(fig, "06_critic_vs_sales.png",
                   "Higher critic scores correlate with better sales (positive slope).")

    # ------------------------------------------------------------------
    # Chart 7 – Top-15 publisher average sales (bar)
    # ------------------------------------------------------------------

    def plot_publisher_power(self):
        pub_avg = (
            self.df.groupby("Publisher")["Global_Sales"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(pub_avg))]
        bars = ax.bar(pub_avg.index, pub_avg.values, color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
        ax.set_ylabel("Average Global Sales (million copies)")
        ax.set_title("Top 15 Publishers by Average Global Sales per Game")
        ax.set_xticks(range(len(pub_avg)))
        ax.set_xticklabels(pub_avg.index, rotation=45, ha='right')  
        plt.tight_layout()                                            
        self._save(fig, "07_publisher_power.png",
                "Nintendo leads by a wide margin in average sales per title.")

    # ------------------------------------------------------------------
    # Chart 8 – Confusion matrix heatmap (called from ModelTrainer)
    # ------------------------------------------------------------------

    def plot_confusion_matrix(self, cm: np.ndarray):
        """
        cm: 2×2 numpy confusion matrix  [[TN, FP], [FN, TP]]
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        labels = ["No Hit", "Hit"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Model Confusion Matrix")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight="bold")

        self._save(fig, "08_confusion_matrix.png",
                   "Confusion matrix shows model performance on held-out test set.")

    # ------------------------------------------------------------------
    # Convenience: run all non-model charts at once
    # ------------------------------------------------------------------

    def plot_all_data_charts(self):
        self.plot_top10_games()
        self.plot_genre_sales()
        self.plot_platform_sales()
        self.plot_year_trend()
        self.plot_regional_comparison()
        self.plot_critic_vs_sales()
        self.plot_publisher_power()
        print("[Visualizer] All 7 data charts saved.")

    # ------------------------------------------------------------------
    # Chart 9 – Model comparison: before vs after (grouped bar chart)
    # ------------------------------------------------------------------

    def plot_model_comparison(self, new_results: dict, baseline_results=None):
        """Plot measured metrics; compare only when a baseline is supplied.

        Both dictionaries use accuracy, precision, recall, f1, and auc in [0, 1].
        The caller is responsible for comparable datasets and evaluation splits.
        """
        keys = ["accuracy", "precision", "recall", "f1", "auc"]
        labels = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
        measured = np.array([new_results[key] for key in keys], dtype=float)
        if not np.isfinite(measured).all() or ((measured < 0) | (measured > 1)).any():
            raise ValueError("Metrics must be finite values between zero and one.")
        baseline = None
        if baseline_results is not None:
            baseline = np.array([baseline_results[key] for key in keys], dtype=float)
            if not np.isfinite(baseline).all() or ((baseline < 0) | (baseline > 1)).any():
                raise ValueError("Baseline metrics must be finite values between zero and one.")

        fig, ax = plt.subplots(figsize=(11, 6))
        positions = np.arange(len(keys))
        width = 0.35 if baseline is not None else 0.6
        if baseline is not None:
            bars = ax.bar(positions - width / 2, baseline * 100, width,
                          label="Supplied baseline", color="#2196F3")
            ax.bar_label(bars, fmt="%.1f%%", padding=3)
        offset = width / 2 if baseline is not None else 0
        bars = ax.bar(positions + offset, measured * 100, width,
                      label="Current run", color="#4CAF50")
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_xticks(positions, labels)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 110)
        ax.set_title("Measured model performance")
        ax.legend()
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        self._save(fig, "09_model_comparison.png", "Metrics supplied by the current evaluation.")
