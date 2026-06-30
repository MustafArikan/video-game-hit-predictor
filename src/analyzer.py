"""
analyzer.py
-----------
DataAnalyzer class:
  - Computes descriptive statistics with NumPy
  - Builds a correlation matrix
  - Produces genre-level and platform-level groupby summaries
  - Saves a text report to outputs/reports/
"""

import numpy as np
import pandas as pd
import os


class DataAnalyzer:
    """Numerical analysis of the video game dataset using NumPy and Pandas."""

    def __init__(self, df: pd.DataFrame, report_dir: str = "outputs/reports"):
        # df must be the cleaned (human-readable) dataframe, not the encoded one
        self.df = df
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

        # Columns we will do NumPy statistics on
        self.numeric_cols = ["Critic_Score", "User_Score", "Global_Sales"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> dict:
        """Run all analyses and save report. Returns summary dict."""
        stats = self.compute_statistics()
        corr = self.correlation_matrix()
        genre_summary = self.genre_groupby()
        platform_summary = self.platform_groupby()

        self._save_report(stats, corr, genre_summary, platform_summary)
        return {
            "statistics": stats,
            "correlation": corr,
            "genre_summary": genre_summary,
            "platform_summary": platform_summary,
        }

    def compute_statistics(self) -> dict:
        """
        Compute mean, std, min, max, sum for key numeric columns using NumPy.
        Also converts a Pandas column to a NumPy array to satisfy the requirement.
        """
        stats = {}
        for col in self.numeric_cols:
            # Explicit Pandas → NumPy array conversion
            arr: np.ndarray = self.df[col].dropna().to_numpy()
            stats[col] = {
                "mean":   float(np.mean(arr)),
                "std":    float(np.std(arr)),
                "min":    float(np.min(arr)),
                "max":    float(np.max(arr)),
                "sum":    float(np.sum(arr)),
                "median": float(np.median(arr)),
            }
        print("[DataAnalyzer] NumPy statistics computed.")
        return stats

    def correlation_matrix(self) -> pd.DataFrame:
        """Pearson correlation matrix for numeric columns."""
        corr = self.df[self.numeric_cols].corr()
        print("[DataAnalyzer] Correlation matrix computed.")
        return corr

    def genre_groupby(self) -> pd.DataFrame:
        """Average Global_Sales per genre, sorted descending."""
        summary = (
            self.df.groupby("Genre")["Global_Sales"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "avg_sales", "sum": "total_sales", "count": "num_games"})
            .sort_values("avg_sales", ascending=False)
        )
        return summary

    def platform_groupby(self) -> pd.DataFrame:
        """Total Global_Sales per platform, sorted descending."""
        summary = (
            self.df.groupby("Platform")["Global_Sales"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "avg_sales", "sum": "total_sales", "count": "num_games"})
            .sort_values("total_sales", ascending=False)
        )
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_report(self, stats, corr, genre_summary, platform_summary):
        """Write a plain-text analysis report."""
        path = os.path.join(self.report_dir, "analysis_report.txt")
        lines = []

        lines.append("=" * 60)
        lines.append("  VIDEO GAME SALES – ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Descriptive statistics
        lines.append("--- NumPy Descriptive Statistics ---")
        for col, s in stats.items():
            lines.append(f"\n{col}:")
            for k, v in s.items():
                lines.append(f"  {k:<8}: {v:.4f}")

        # Correlation
        lines.append("\n--- Pearson Correlation Matrix ---")
        lines.append(corr.to_string())

        # Genre
        lines.append("\n--- Genre Avg Global Sales (M copies) ---")
        lines.append(genre_summary.to_string())

        # Platform
        lines.append("\n--- Platform Total Global Sales (M copies, top 15) ---")
        lines.append(platform_summary.head(15).to_string())

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[DataAnalyzer] Report saved → {path}")
