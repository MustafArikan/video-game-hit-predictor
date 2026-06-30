"""
model_trainer.py
----------------
ModelTrainer class handles model building, training, and evaluation.

Two models are trained and combined into an ensemble:
    1. Keras Dense Neural Network (DNN)
       - Required by the project spec (TensorFlow/Keras)
       - Good at learning non-linear interactions between features

    2. XGBoost Classifier
       - Gradient-boosted decision trees
       - Excellent for tabular data with mixed types
       - Tuned with RandomizedSearchCV (hyperparameter optimization)

Ensemble prediction:
    final_probability = 0.5 * DNN_probability + 0.5 * XGBoost_probability
    Averaging two diverse models reduces variance and improves F1.

Training safeguards applied:
    - StandardScaler: normalize feature values (fit only on train set)
    - SMOTE: oversample minority class in training set to fix imbalance
    - EarlyStopping: stop DNN training when validation loss stops improving
    - ReduceLROnPlateau: lower DNN learning rate when stuck on a plateau
    - RandomizedSearchCV: find best XGBoost hyperparameters automatically
    - Threshold optimization: find the prediction cutoff that maximizes F1
"""

import os
import warnings
import numpy as np

# Suppress TensorFlow C++ info/warning messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE


class ModelTrainer:
    """
    Trains a DNN + XGBoost ensemble to predict whether a video game will be a hit.

    Usage:
        trainer = ModelTrainer(X, y)
        results = trainer.train()   # returns dict with all metrics
        cm      = trainer.get_confusion_matrix()
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 report_dir: str = "outputs/reports"):
        # Store the feature matrix and target vector
        self.X = X
        self.y = y

        # Directory where the text report will be saved
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

        # These will be filled during train()
        self.dnn_model    = None   # Keras Sequential model
        self.xgb_model    = None   # XGBoost classifier (after hyperparameter tuning)
        self.scaler       = StandardScaler()
        self.history      = None   # Keras training history (loss per epoch)
        self.threshold    = 0.5    # decision cutoff, will be optimized on val set

        # Train / val / test splits (set in _split)
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Run the full training pipeline and return a dictionary of metrics.
        Steps: split -> scale -> SMOTE -> DNN -> XGBoost+tuning -> threshold -> evaluate
        """
        self._split()
        self._scale()
        self._apply_smote()
        self._build_and_fit_dnn()
        self._tune_and_fit_xgboost()
        self._find_best_threshold()
        results = self._evaluate()
        self._save_report(results)
        return results

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Return the 2x2 confusion matrix for the ensemble on the test set.
        Must call train() before this.
        """
        if self.dnn_model is None or self.xgb_model is None:
            raise RuntimeError("Call train() first.")

        # Get ensemble probabilities on the test set
        probs = self._ensemble_predict(self.X_test)

        # Apply the optimized threshold to convert probabilities to 0/1 labels
        y_pred = (probs >= self.threshold).astype(int)
        return confusion_matrix(self.y_test, y_pred)

    # ── Private pipeline steps ─────────────────────────────────────────────────

    def _split(self):
        """
        Split data into train (70%), validation (15%), and test (15%) sets.

        Stratify keeps the class ratio (~12% hits) the same in all three splits.
        Without stratify, one split could have very few hit examples by chance.
        """
        # First cut: separate test set (15%)
        X_tmp, self.X_test, y_tmp, self.y_test = train_test_split(
            self.X, self.y,
            test_size=0.15,
            random_state=42,
            stratify=self.y       # keep hit ratio consistent
        )

        # Second cut: split remaining 85% into train and validation
        # 0.15/0.85 ≈ 0.176 so that validation ends up as 15% of the total
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_tmp, y_tmp,
            test_size=0.15 / 0.85,
            random_state=42,
            stratify=y_tmp
        )

        print(f"[ModelTrainer] Split: train={len(self.X_train):,} | "
              f"val={len(self.X_val):,} | test={len(self.X_test):,}")

    def _scale(self):
        """
        Normalize all feature values to have mean=0 and std=1.

        Why fit only on train?
        If we fit the scaler on val+test too, we expose the model to statistics
        from data it has never 'seen' during training -- data leakage.
        In production, we would only ever have the training distribution available.
        """
        # fit_transform: compute mean/std from train data AND scale it
        self.X_train = self.scaler.fit_transform(self.X_train)

        # transform only: use the SAME mean/std from training to scale val and test
        self.X_val   = self.scaler.transform(self.X_val)
        self.X_test  = self.scaler.transform(self.X_test)

        print("[ModelTrainer] Features scaled with StandardScaler (fit on train only).")

    def _apply_smote(self):
        """
        Balance the training set using SMOTE (Synthetic Minority Oversampling).

        The dataset has ~12% hits and ~88% no-hits.
        Without balancing, the model can score 88% accuracy by always predicting 0.
        SMOTE creates synthetic 'hit' examples by interpolating between existing hits
        using k-nearest neighbors, until the class ratio is 50/50.

        Applied ONLY to training data -- val and test stay at the real 12% ratio
        so that our evaluation reflects real-world performance.
        """
        before_rate = self.y_train.mean()

        # k_neighbors=5 means each synthetic point is generated between 5 real neighbors
        smote = SMOTE(random_state=42, k_neighbors=5)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

        after_rate = self.y_train.mean()
        print(f"[ModelTrainer] SMOTE: {len(self.X_train):,} training samples | "
              f"hit rate {before_rate:.1%} -> {after_rate:.1%}")

    def _build_and_fit_dnn(self):
        """
        Build and train the Keras Dense Neural Network.

        Architecture:
            Input(12) -> Dense(128, ReLU) -> BatchNorm -> Dropout(0.4)
                      -> Dense(64,  ReLU) -> BatchNorm -> Dropout(0.3)
                      -> Dense(32,  ReLU) -> Dropout(0.2)
                      -> Dense(1,   Sigmoid)

        Additions compared to a bare sequential model:
          - BatchNormalization: normalizes activations between layers,
            speeds up training and allows higher learning rates
          - L2 regularization (1e-4): penalizes large weights to reduce overfitting
          - clipnorm=1.0: clips gradient norm to prevent NaN loss
          - EarlyStopping: stops training when val_loss stops improving
          - ReduceLROnPlateau: halves learning rate when training gets stuck
        """
        n_features = self.X_train.shape[1]   # number of input features (12)

        self.dnn_model = keras.Sequential([
            layers.Input(shape=(n_features,)),

            # Layer 1: 128 neurons with ReLU activation
            layers.Dense(128, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),   # normalize outputs of this layer
            layers.Dropout(0.4),          # randomly turn off 40% of neurons during training

            # Layer 2: 64 neurons
            layers.Dense(64, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Layer 3: 32 neurons
            layers.Dense(32, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.2),

            # Output: single neuron with sigmoid -> outputs probability 0 to 1
            layers.Dense(1, activation="sigmoid"),

        ], name="VideoGame_HitPredictor_DNN")

        # Adam optimizer with gradient clipping to prevent NaN loss
        self.dnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        self.dnn_model.summary()

        # ── Callbacks ─────────────────────────────────────────────────────────
        callbacks = [
            # Stop training if val_loss does not improve for 12 consecutive epochs
            # restore_best_weights=True: go back to the epoch with the best val_loss
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
                verbose=1,
            ),
            # Reduce learning rate by half if val_loss is stuck for 6 epochs
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=6,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        # ── Train ──────────────────────────────────────────────────────────────
        print("\n[ModelTrainer] Training DNN...")
        self.history = self.dnn_model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=150,         # maximum epochs (EarlyStopping will usually stop earlier)
            batch_size=64,
            callbacks=callbacks,
            verbose=1,
        )
        print(f"[ModelTrainer] DNN training finished after "
              f"{len(self.history.history['loss'])} epochs.")

    def _tune_and_fit_xgboost(self):
        """
        Tune XGBoost hyperparameters using RandomizedSearchCV, then train.

        Why RandomizedSearchCV instead of GridSearchCV?
          GridSearchCV tests EVERY combination of parameters (slow).
          RandomizedSearchCV randomly samples n_iter combinations (faster).
          With n_iter=30, we test 30 random combinations out of hundreds possible.

        Key XGBoost hyperparameters we tune:
          n_estimators   : number of trees (more = stronger, slower)
          max_depth       : how deep each tree grows (deeper = more complex)
          learning_rate   : how much each tree contributes (smaller = more stable)
          subsample        : fraction of rows each tree sees (prevents overfitting)
          colsample_bytree : fraction of features each tree sees (prevents overfitting)
          min_child_weight : minimum samples needed to create a leaf node
          gamma            : minimum loss reduction to make a split
        """
        print("\n[ModelTrainer] Tuning XGBoost with RandomizedSearchCV (30 iterations)...")

        # Define the hyperparameter search space
        # RandomizedSearchCV will randomly pick combinations from these lists
        param_dist = {
            "n_estimators":      [200, 300, 400, 500],
            "max_depth":         [3, 4, 5, 6, 7],
            "learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2],
            "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight":  [1, 3, 5, 7],
            "gamma":             [0, 0.1, 0.2, 0.3],
        }

        # scale_pos_weight: handle class imbalance
        # Ratio of negative to positive samples in the ORIGINAL (pre-SMOTE) training set
        # This tells XGBoost to pay more attention to hit examples
        neg_count = int((self.y_train == 0).sum())
        pos_count = int((self.y_train == 1).sum())
        # After SMOTE they are equal (1.0), so scale_pos_weight = 1
        scale_pos_weight = neg_count / max(pos_count, 1)

        # Base XGBoost classifier
        base_xgb = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",   # evaluation metric during training
            random_state=42,
            verbosity=0,             # suppress XGBoost verbose output
            n_jobs=-1,               # use all CPU cores
        )

        # RandomizedSearchCV wraps the base model with cross-validation
        # cv=3: 3-fold cross-validation for each parameter combination
        # scoring="f1": optimize for F1 score (better than accuracy for imbalanced data)
        # n_iter=30: test 30 random parameter combinations
        search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist,
            n_iter=30,
            scoring="f1",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        # Fit the search on the SMOTE-balanced training data
        search.fit(self.X_train, self.y_train)

        # The best estimator is the XGBoost model with the best hyperparameters
        self.xgb_model = search.best_estimator_

        print(f"[ModelTrainer] Best XGBoost params: {search.best_params_}")
        print(f"[ModelTrainer] Best CV F1 score: {search.best_score_:.4f}")

    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute ensemble probability by averaging DNN and XGBoost outputs.

        Both models output a probability between 0 and 1.
        Averaging reduces the error of each individual model:
          - DNN captures non-linear feature interactions
          - XGBoost is robust to outliers and handles tree-based patterns well
          Together they cover each other's blind spots.
        """
        # DNN: predict() returns shape (n, 1), flatten to (n,)
        dnn_probs = self.dnn_model.predict(X, verbose=0).ravel()

        # XGBoost: predict_proba() returns shape (n, 2), take column 1 (probability of hit)
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]

        # Simple average: weight each model equally
        ensemble_probs = (dnn_probs + xgb_probs) / 2.0
        return ensemble_probs

    def _find_best_threshold(self):
        """
        Find the decision threshold that maximizes F1 on the validation set.

        Default threshold is 0.5, but this is arbitrary.
        With imbalanced data and after SMOTE, the optimal cutoff often shifts.
        We scan all thresholds from 0.10 to 0.90 in steps of 0.01,
        compute F1 at each step, and keep the threshold with the best F1.

        This is done on the VALIDATION set -- the test set is never touched here.
        """
        # Get ensemble probabilities on the validation set
        val_probs = self._ensemble_predict(self.X_val)

        best_threshold = 0.5
        best_f1        = 0.0

        # Try 80 different thresholds between 0.10 and 0.90
        for t in np.arange(0.10, 0.91, 0.01):
            # Convert probabilities to 0/1 predictions using this threshold
            y_pred_t = (val_probs >= t).astype(int)

            # Compute precision and recall to get F1
            p = precision_score(self.y_val, y_pred_t, zero_division=0)
            r = recall_score(self.y_val, y_pred_t, zero_division=0)

            # F1 = harmonic mean of precision and recall
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
                if f1 > best_f1:
                    best_f1        = f1
                    best_threshold = float(t)

        self.threshold = best_threshold
        print(f"[ModelTrainer] Optimal threshold: {self.threshold:.2f}  "
              f"(validation F1 = {best_f1:.4f})")

    def _evaluate(self) -> dict:
        """
        Evaluate the ensemble on the held-out test set using the optimized threshold.
        The test set has NEVER been used in any previous step -- honest evaluation.
        """
        # Get final predictions on the test set
        test_probs = self._ensemble_predict(self.X_test)
        y_pred     = (test_probs >= self.threshold).astype(int)

        # Compute all metrics
        acc  = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec  = recall_score(self.y_test, y_pred, zero_division=0)
        f1   = f1_score(self.y_test, y_pred, zero_division=0)

        # AUC: area under the ROC curve -- threshold-independent measure of ranking quality
        # 1.0 = perfect, 0.5 = random guessing
        auc  = roc_auc_score(self.y_test, test_probs)

        # Print results table
        print("\n" + "=" * 55)
        print("  ENSEMBLE TEST SET RESULTS  (DNN + XGBoost)")
        print("=" * 55)
        print(f"  Threshold : {self.threshold:.2f}  (optimized on val set)")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  Precision : {prec*100:.2f}%")
        print(f"  Recall    : {rec*100:.2f}%")
        print(f"  F1 Score  : {f1*100:.2f}%")
        print(f"  AUC-ROC   : {auc*100:.2f}%")
        print("=" * 55)
        print()
        print(classification_report(self.y_test, y_pred,
                                    target_names=["No Hit (0)", "Hit (1)"]))

        return {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "auc":       auc,
        }

    def _save_report(self, results: dict):
        """Write a summary of the model and its results to a text file."""
        path = os.path.join(self.report_dir, "model_report.txt")

        # Build the report as a list of strings, then join with newlines
        lines = [
            "=" * 55,
            "  MODEL REPORT: DNN + XGBoost Ensemble",
            "=" * 55,
            "",
            "── DNN Architecture ────────────────────────────────",
            "  Input(12)",
            "  -> Dense(128, ReLU) + BatchNorm + Dropout(0.4)",
            "  -> Dense(64,  ReLU) + BatchNorm + Dropout(0.3)",
            "  -> Dense(32,  ReLU) + Dropout(0.2)",
            "  -> Dense(1,   Sigmoid)",
            "",
            "── XGBoost ─────────────────────────────────────────",
            f"  Best params: {self.xgb_model.get_params()}",
            "",
            "── Improvements Applied ────────────────────────────",
            "  1. fillna(-1) + Has_Critic_Score / Has_User_Score flags",
            "  2. Target encoding: Publisher, Genre, Platform",
            "  3. SMOTE oversampling on train set (12% -> 50% hit rate)",
            "  4. XGBoost tuned with RandomizedSearchCV (30 iters, cv=3)",
            "  5. Threshold optimized on val set for max F1",
            "  6. Ensemble: DNN + XGBoost probability average",
            "",
            "── Test Set Metrics ────────────────────────────────",
            f"  Threshold : {self.threshold:.2f}",
            f"  Accuracy  : {results['accuracy']*100:.2f}%",
            f"  Precision : {results['precision']*100:.2f}%",
            f"  Recall    : {results['recall']*100:.2f}%",
            f"  F1 Score  : {results['f1']*100:.2f}%",
            f"  AUC-ROC   : {results['auc']*100:.2f}%",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[ModelTrainer] Report saved -> {path}")
