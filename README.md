# Video Game Hit Predictor

Explore historical game sales and train a Keras/XGBoost ensemble to classify games whose global sales exceed one million copies.

The project combines data analysis, chart generation, feature engineering, and experimental classification. Review scores and sales-derived features describe historical data; the current pipeline is not a validated pre-release forecasting system.

## Key Features

- CSV cleaning, score-missingness flags, and categorical feature engineering.
- Descriptive reports and seven exploratory charts.
- A 50/50 average of Keras DNN and XGBoost prediction probabilities.
- Training-set scaling, SMOTE, randomized XGBoost parameter search, and validation-set threshold selection.
- Test metrics, a confusion matrix, and a chart of metrics supplied by the evaluation.
- Optional baseline comparison only when the caller supplies baseline measurements.

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, TensorFlow/Keras, and XGBoost. See [requirements.txt](requirements.txt).

## Architecture & Project Structure

```text
.
├── data/
│   └── Video_Games_Sales_as_at_22_Dec_2016.csv
├── src/
│   ├── main.py           # Pipeline orchestration and repository-relative paths
│   ├── data_loader.py    # Cleaning, target creation, feature engineering
│   ├── analyzer.py       # Descriptive statistics and reports
│   ├── model_trainer.py  # Splits, scaling, SMOTE, models, threshold, metrics
│   └── visualizer.py     # Exploratory and evaluation charts
├── outputs/              # Historical reports/charts; overwritten on reruns
├── tests/                # Chart provenance regression tests
├── requirements.txt
└── LICENSE
```

`Global_Sales > 1.0` defines `Hit`; sales values are in millions. The model uses 12 numeric/encoded features. Training allocates 70%/15%/15% to training/validation/test with seed 42.

**Evaluation limitations:** category target encodings and aggregate features are calculated on the complete dataset before splitting. That exposes held-out labels/statistics to feature construction. SMOTE is also applied before the inner randomized-search folds, so inner CV can include related synthetic observations. Scores and historical comparison images must not be presented as independent generalization evidence. Repair these boundaries and retrain before making performance claims.

## Getting Started

Use Python 3.11 and an isolated environment. TensorFlow/XGBoost support depends on your platform; GPU configuration is separate from these commands.

```bash
git clone https://github.com/MustafArikan/video-game-hit-predictor.git
cd video-game-hit-predictor
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -r requirements.txt
```

## Configuration

No credentials or mandatory environment variables are required. Data/report/chart paths are based on the location of `src/main.py`. Training parameters, the sales threshold, and feature names are currently source-code constants.

## Usage

```bash
python src/main.py
```

This runs cleaning, analysis, chart export, training, hyperparameter search, threshold selection, and evaluation. Training may take substantial time and use all CPU cores. Reports are written under `outputs/reports/`; charts go to `outputs/graphs/`.

![Historical regional sales comparison](outputs/graphs/05_regional_comparison.png)

The performance-chart method retains its existing name for compatibility. `plot_model_comparison(results)` plots only the supplied run. An optional second dictionary adds an explicitly supplied baseline; the code no longer embeds unverified baseline numbers or fixed improvement claims. Existing committed comparison images predate this correction.

## Testing

```bash
python -m unittest discover -s tests
```

Tests verify that charts use supplied measurements and do not invent a baseline. They exercise visualization without training TensorFlow models. Full experiment validation requires running the pipeline after correcting the data-leakage boundaries.

## Development & Quality

```bash
python -m compileall -q src tests
python -m pip check
```

No CI workflow or production inference service is configured. Dependency ranges are not a lockfile. Keep historical artifacts distinct from new runs and record versions, seeds, splits, and dataset provenance with any new result.

## Roadmap

- Split before feature learning; fit target encodings using out-of-fold training labels.
- Fit resampling and preprocessing inside each model-selection fold.
- Add a reproducible baseline evaluated on the identical untouched holdout.
- Separate post-release analysis from any future pre-release prediction task.

## Contributing

Include reproducible inputs and evaluation methodology with model changes. Run regression tests, preserve existing interfaces where practical, and avoid publishing performance comparisons without traceable measurements.

## License

Code is distributed under the [MIT License](LICENSE). Check the dataset's source terms before redistribution.

## Author

[Mustafa Arıkan](https://github.com/MustafArikan)
