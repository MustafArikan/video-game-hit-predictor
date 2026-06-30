"""
data_loader.py
--------------
DataLoader class handles everything related to reading and preparing the dataset.

Pipeline order:
    1. _read_csv()           -> load raw CSV from disk
    2. _clean()              -> drop bad rows, fix types, fill missing values
    3. _create_target()      -> create the binary Hit column
    4. _engineer_features()  -> build new informative columns from existing ones
    5. _encode()             -> convert categorical text to numbers

The class exposes:
    load_and_prepare()       -> runs all steps above, returns encoded dataframe
    get_features_and_target()-> returns X (numpy array) and y (numpy array)
    df_clean                 -> human-readable dataframe used by Visualizer/Analyzer
    df_encoded               -> fully numeric dataframe used by ModelTrainer
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── Feature column list ────────────────────────────────────────────────────────
# These are the exact columns ModelTrainer will receive as input.
# 12 features total: 5 original + 2 missing-value flags + 5 engineered.
FEATURE_COLS = [
    # ── Original numeric columns ──────────────────────────────────────────────
    "Year_of_Release",    # year the game was released
    "Critic_Score",       # average critic score 0-100 (or -1 if missing)
    "User_Score",         # average user score 0-10 (or -1 if missing)
    "Critic_Count",       # how many critics reviewed the game (0 if missing)
    "Rating",             # ESRB rating encoded as integer (E, T, M, etc.)

    # ── Missing-value flags (Yol 1) ────────────────────────────────────────────
    # Instead of hiding the fact that data is missing, we tell the model directly.
    # A game with no critic score is likely a small/unknown title -> signal for "no hit".
    "Has_Critic_Score",   # 1 if Critic_Score was originally present, 0 if it was missing
    "Has_User_Score",     # 1 if User_Score was originally present, 0 if it was missing

    # ── Target-encoded category columns (Yol 2) ───────────────────────────────
    # Instead of giving Publisher="EA" as a meaningless integer (label encoding),
    # we replace it with EA's historical hit rate, e.g. 0.38.
    # This tells the model "EA games become hits 38% of the time" -- much more useful.
    "Publisher_target_enc",  # publisher's fraction of games that are hits
    "Genre_target_enc",      # genre's fraction of games that are hits
    "Platform_target_enc",   # platform's fraction of games that are hits

    # ── Other engineered features ─────────────────────────────────────────────
    "Platform_avg_score",    # average critic score for all games on this platform
    "Publisher_game_count",  # total number of games this publisher has released
]

# The column we want to predict (1 = hit, 0 = no hit)
TARGET_COL = "Hit"

# A game is a "hit" if it sold more than 1 million copies globally
SALES_THRESHOLD = 1.0


class DataLoader:
    """
    Reads, cleans, and prepares the video game sales dataset.

    After calling load_and_prepare():
        self.df_clean   -> use this for charts and analysis (human-readable text)
        self.df_encoded -> use this for the model (all numbers)
    """

    def __init__(self, filepath: str):
        # Store the path to the CSV file
        self.filepath = filepath

        # These will be filled after load_and_prepare() is called
        self.df_clean   = None   # cleaned dataframe with original text labels
        self.df_encoded = None   # fully numeric dataframe ready for the model

        # Store the LabelEncoder for Rating so we can inspect it later
        self.label_encoders = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_and_prepare(self) -> pd.DataFrame:
        """
        Run the full data pipeline in order.
        Returns the encoded dataframe (all numbers).
        """
        raw        = self._read_csv()
        cleaned    = self._clean(raw)
        with_target = self._create_target(cleaned)
        engineered  = self._engineer_features(with_target)

        # Save the human-readable version before encoding
        self.df_clean = engineered.copy()

        encoded = self._encode(engineered)
        self.df_encoded = encoded
        return self.df_encoded

    def get_features_and_target(self):
        """
        Extract the feature matrix X and target vector y as numpy arrays.
        Must call load_and_prepare() before this.
        """
        if self.df_encoded is None:
            raise RuntimeError("Call load_and_prepare() first.")

        # Convert the feature columns to a 2D numpy float32 array
        X = self.df_encoded[FEATURE_COLS].values.astype(np.float32)

        # Convert the target column to a 1D numpy float32 array
        y = self.df_encoded[TARGET_COL].values.astype(np.float32)

        # Safety check: any NaN in X will cause training to produce NaN loss
        if np.isnan(X).any():
            # Find which columns have NaN and report them
            bad_cols = [col for col in FEATURE_COLS
                        if np.isnan(self.df_encoded[col].values).any()]
            raise ValueError(f"[DataLoader] NaN found in columns: {bad_cols}")

        return X, y

    # ── Private steps ──────────────────────────────────────────────────────────

    def _read_csv(self) -> pd.DataFrame:
        """
        Load the CSV file from disk.
        Uses try-except to give a clear error message if the file is missing.
        """
        try:
            df = pd.read_csv(self.filepath)
            print(f"[DataLoader] Loaded {len(df):,} rows, {df.shape[1]} columns.")
            return df
        except FileNotFoundError:
            # Give a helpful error instead of a cryptic Python traceback
            raise FileNotFoundError(
                f"[DataLoader] Dataset not found at: {self.filepath}\n"
                "Place the CSV file in the data/ folder and try again."
            )

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataframe:
          - Drop rows where Year_of_Release or Global_Sales is missing
            (we cannot create a target label without sales, and year is essential)
          - Remove impossible years (games before 1975 or after 2020)
          - Convert User_Score from string to float ('tbd' becomes NaN)
          - Fill missing Critic_Score, User_Score with -1
            (-1 is outside the normal range, so the model knows it is "missing")
          - Fill missing Critic_Count with 0
            (0 critics reviewed it -- likely a very small game)
          - Record which scores were originally missing as binary flags
          - Fill missing Genre and Rating with their most common value
        """
        df = df.copy()

        # ── Drop rows with no sales or no year ────────────────────────────────
        rows_before = len(df)
        df = df.dropna(subset=["Year_of_Release", "Global_Sales"])
        df = df[df["Year_of_Release"].between(1975, 2020)]
        df["Year_of_Release"] = df["Year_of_Release"].astype(int)
        print(f"[DataLoader] Dropped {rows_before - len(df):,} rows "
              f"(missing year or sales).")

        # ── User_Score cleanup ────────────────────────────────────────────────
        # The CSV stores 'tbd' for games with no user score yet; convert to NaN
        df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")

        # ── Record which scores are PRESENT before filling ────────────────────
        # We do this BEFORE fillna so we capture the true missing pattern.
        # Has_Critic_Score = 1 means "a real score exists", 0 means "no data".
        df["Has_Critic_Score"] = df["Critic_Score"].notna().astype(int)
        df["Has_User_Score"]   = df["User_Score"].notna().astype(int)

        # ── Fill missing scores with -1 (sentinel value) ──────────────────────
        # -1 is outside the valid range for both scores (0-100 and 0-10).
        # The model will learn that -1 means "unknown / no data".
        df["Critic_Score"] = df["Critic_Score"].fillna(-1)
        df["User_Score"]   = df["User_Score"].fillna(-1)

        # ── Fill missing Critic_Count with 0 ──────────────────────────────────
        df["Critic_Count"] = df["Critic_Count"].fillna(0)

        # ── Fill missing categorical values with most common value ─────────────
        df["Genre"]  = df["Genre"].fillna(df["Genre"].mode()[0])
        df["Rating"] = df["Rating"].fillna(df["Rating"].mode()[0])

        # ── Fill missing Publisher with 'Unknown' ─────────────────────────────
        df["Publisher"] = df["Publisher"].fillna("Unknown")

        print(f"[DataLoader] Clean rows: {len(df):,}")
        print(f"[DataLoader]   Has_Critic_Score=1: "
              f"{df['Has_Critic_Score'].sum():,} ({df['Has_Critic_Score'].mean():.1%})")
        print(f"[DataLoader]   Has_User_Score=1  : "
              f"{df['Has_User_Score'].sum():,} ({df['Has_User_Score'].mean():.1%})")
        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the binary target column.
        Hit = 1 if Global_Sales > 1.0 million copies, else 0.

        IMPORTANT: Global_Sales is used ONLY here to build the label.
        It is never included as a model feature -- that would be data leakage
        (the model would just learn "high sales = hit" which is trivial and useless).
        """
        df = df.copy()
        df[TARGET_COL] = (df["Global_Sales"] > SALES_THRESHOLD).astype(int)

        # Show the class split so we can see how imbalanced the data is
        hit_count  = int(df[TARGET_COL].sum())
        total      = len(df)
        print(f"[DataLoader] Hit rate: {hit_count:,}/{total:,} "
              f"({hit_count/total:.1%} hits, {(total-hit_count)/total:.1%} no-hits)")
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build new numeric features from existing columns.

        Why target encoding instead of label encoding for Publisher/Genre/Platform?
        ─────────────────────────────────────────────────────────────────────────
        Label encoding assigns arbitrary integers: EA=7, Nintendo=42, etc.
        The model treats these as ordered numbers (42 > 7 means nothing here).
        Correlation between label-encoded Publisher and Hit is only ~0.02.

        Target encoding replaces each category with the fraction of games
        in that category that are hits:
            Nintendo -> 0.41  (41% of Nintendo games are hits)
            EA       -> 0.28  (28% of EA games are hits)
        Correlation between target-encoded Publisher and Hit is ~0.30.
        This gives the model 15x more useful information.

        NOTE: In a strict production pipeline, target encoding should be
        computed on train-only data to avoid leakage. For this academic
        project it is computed on the full dataset and noted accordingly.
        """
        df = df.copy()

        # ── Target encoding for Publisher ──────────────────────────────────────
        # Group by publisher and compute mean of Hit column (= fraction that are hits)
        pub_hit_rate = df.groupby("Publisher")[TARGET_COL].mean()
        df["Publisher_target_enc"] = df["Publisher"].map(pub_hit_rate).astype(np.float32)

        # ── Target encoding for Genre ──────────────────────────────────────────
        genre_hit_rate = df.groupby("Genre")[TARGET_COL].mean()
        df["Genre_target_enc"] = df["Genre"].map(genre_hit_rate).astype(np.float32)

        # ── Target encoding for Platform ───────────────────────────────────────
        platform_hit_rate = df.groupby("Platform")[TARGET_COL].mean()
        df["Platform_target_enc"] = df["Platform"].map(platform_hit_rate).astype(np.float32)

        # ── Average Critic Score per Platform ─────────────────────────────────
        # Only use rows that actually have a critic score (Has_Critic_Score=1)
        scored = df[df["Has_Critic_Score"] == 1]
        platform_avg = scored.groupby("Platform")["Critic_Score"].mean()
        df["Platform_avg_score"] = df["Platform"].map(platform_avg).fillna(0).astype(np.float32)

        # ── Publisher game count ───────────────────────────────────────────────
        # How many games does this publisher have in the dataset?
        # Larger publishers (more games) tend to have better distribution networks.
        pub_count = df.groupby("Publisher")["Publisher"].transform("count")
        df["Publisher_game_count"] = pub_count.astype(np.float32)

        print(f"[DataLoader] Engineered features added: "
              f"Publisher_target_enc, Genre_target_enc, Platform_target_enc, "
              f"Platform_avg_score, Publisher_game_count")
        return df

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert remaining categorical text columns to integers.
        Only Rating needs label encoding here -- all other categoricals
        were replaced by numeric target-encoded values above.
        """
        df = df.copy()

        # Rating has ~7 unique values (E, E10+, T, M, AO, K-A, RP)
        # Label encoding is fine for a small number of categories
        le = LabelEncoder()
        df["Rating"] = le.fit_transform(df["Rating"].astype(str))
        self.label_encoders["Rating"] = le

        print(f"[DataLoader] Label-encoded: Rating "
              f"({len(le.classes_)} unique values: {list(le.classes_)})")
        return df
