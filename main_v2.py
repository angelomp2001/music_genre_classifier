# libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, Conv1D,
    GlobalAveragePooling1D, Multiply, Lambda, Concatenate, Activation, Masking)

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K


# EDA
features_3_sec_path = 'Data/features_3_sec.csv' # C:\Users\apass\OneDrive\Documents\VS code\Music_genre_predictor\Data
features_30_sec_path = 'Data/features_30_sec.csv' # C:\Users\apass\OneDrive\Documents\VS code\Music_genre_predictor\Data
df = pd.read_csv(features_3_sec_path) # C:\Users\Angelo\Downloads\GTZAN data set\Data
# df.shape # (9990, 60)
# df.dtypes # filename, label = object. length = int. the rest = float
# df.isna().sum().sum() # np.int64(0)
df_30_sec = pd.read_csv(features_30_sec_path) # C:\Users\Angelo\Downloads\GTZAN data set\Data
df_30_sec.head()
df.head()


# merge dfs
# create index to match index of other df, save as 'common_key'
df = df.copy()
df.rename(columns={'filename': 'file_segment'}, inplace = True)
df["filename"] = (
    df['file_segment']
    .str.replace(r"\.\d+\.wav$", ".wav", regex=True)
)

# move it to loc 0, so I can see it.
df.insert(0, 'filename', df.pop('filename'))

df = (
    df_30_sec
    .merge(df, how = 'left', on = 'filename', suffixes = ('_file', '_seg'))
)

df.insert(1, 'file_segment', df.pop('file_segment'))

TARGET_COL = 'label_seg'
GROUP_COL = 'filename'


# split data function
def build_track_sequences(df, feature_cols, label_col=TARGET_COL, group_col=GROUP_COL):
    """
    Build per-track sequences from a segment-level dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least [group_col, label_col] and feature_cols.
    feature_cols : list of str
        Names of numeric feature columns to use as segment features.
    label_col : str
        Segment-level label column; assumed constant within each track.
    group_col : str
        Column representing the track identifier (e.g., filename).

    Returns
    -------
    X_seq : list of np.ndarray
        Each element has shape (T_i, n_features) for track i.
    y_track : np.ndarray
        Array of track-level labels, one per track.
    df_subset : pd.DataFrame
        One row per track (metadata).
    """
    groups = df[group_col].unique()

    # GLOBAL SCALING FOR SEQUENCE FEATURES (fit on all segments once)
    # You can also pass a pre-fitted scaler instead of fitting here.
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    X_seq = []
    y_track = []
    df_subset_rows = []

    for g in groups:
        dfg = df_scaled[df_scaled[group_col] == g]
        # Each row is one segment; use scaled numeric features
        seq = dfg[feature_cols].to_numpy()  # shape: (T_i, F)
        X_seq.append(seq)

        # Track-level label: assume all segments share same label
        y_track.append(dfg[label_col].iloc[0])

        # Keep one representative row for metadata
        df_subset_rows.append(dfg.iloc[0])

    df_subset = pd.DataFrame(df_subset_rows)

    return X_seq, np.array(y_track), df_subset

# split sequences
def split_sequences(groups, X_seq, y_track, test_size=0.2, random_state=12345):
    """
    Track-level split for sequences.

    Parameters
    ----------
    groups : array-like
        Array of track identifiers, one per sequence.
    X_seq : list of np.ndarray
        List of sequences, length = num_tracks.
    y_track : np.ndarray
        Track-level labels, length = num_tracks.
    test_size : float
        Fraction of tracks to use for test.
    random_state : int
        Random seed.

    Returns
    -------
    X_train_seq, X_test_seq, y_train, y_test, groups_train, groups_test
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_test = next(gss.split(X_seq, y=y_track, groups=groups))

    X_train_seq = [X_seq[i] for i in idx_train]
    X_test_seq = [X_seq[i] for i in idx_test]

    y_train = y_track[idx_train]
    y_test = y_track[idx_test]

    groups = np.asarray(groups)
    groups_train = groups[idx_train]
    groups_test = groups[idx_test]

    return X_train_seq, X_test_seq, y_train, y_test, groups_train, groups_test

# logistic regression classes
class LogisticSegmentModelWithFixedSplit:
    """
    Logistic regression trained at the segment level (3-sec segments),
    evaluated at the track level by aggregating segment probabilities.

    Uses a fixed track-level split: train_tracks vs test_tracks (same as deep model).
    """

    def __init__(self, df, train_tracks, test_tracks,
                 label_col=TARGET_COL, group_col=GROUP_COL):
        self.df = df.copy()
        self.label_col = label_col
        self.group_col = group_col
        self.train_tracks = np.array(train_tracks)
        self.test_tracks = np.array(test_tracks)

        # Subset segments by track split
        self.df_train = self.df[self.df[group_col].isin(self.train_tracks)]
        self.df_test = self.df[self.df[group_col].isin(self.test_tracks)]

        # Numeric features only
        self.X_train = self.df_train.select_dtypes(include=['float64', 'int64'])
        self.X_test = self.df_test.select_dtypes(include=['float64', 'int64'])

        # FEATURE SCALING (fit on train, apply to test)
        self.scaler = MinMaxScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Segment-level labels
        self.y_train = self.df_train[label_col]
        self.y_test = self.df_test[label_col]

        self.model = None

    def fit(self, path=None):
        """
        Train (or load) logistic regression and evaluate at track level.
        Returns a dictionary with overall and per-class accuracy.
        """
        if path is None:
            log_reg = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=2000,
                n_jobs=-1
            )
            log_reg.fit(self.X_train, self.y_train)
            dump(log_reg, "logistic_regression_baseline.joblib")
            self.model = log_reg
        else:
            self.model = load(path)

        # Predict segment-level probabilities on test segments
        proba_segments = self.model.predict_proba(self.X_test)
        classes = self.model.classes_

        proba_df = pd.DataFrame(
            proba_segments,
            index=self.df_test.index,
            columns=classes
        )
        proba_df[self.group_col] = self.df_test[self.group_col].values

        # Aggregate segment probabilities to track level (mean)
        track_proba = proba_df.groupby(self.group_col)[classes].mean()
        y_pred = track_proba.idxmax(axis=1)

        # True track-level labels
        y_true = track_proba.index.map(
            lambda g: self.df.loc[self.df[self.group_col] == g, self.label_col].iloc[0]
        )

        overall_acc = accuracy_score(y_true, y_pred)

        # Per-class accuracy
        per_class_accuracy = {}
        for c in classes:
            mask = (y_true == c)
            total = mask.sum()
            correct = (y_pred[mask] == c).sum() if total > 0 else 0
            per_class_accuracy[c] = correct / total if total > 0 else np.nan

        per_class_df = pd.DataFrame.from_dict(
            per_class_accuracy,
            orient='index',
            columns=['accuracy']
        )

        return {
            "overall accuracy": overall_acc,
            "per class accuracy": per_class_df
        }

# ============================================================
# 4. Segment-Level Logistic Regression with Feature Engineering
#    (Track Aggregation)
# ============================================================

class LogisticSegmentModelWithFE:
    """
    Logistic regression with polynomial feature engineering at the segment level,
    evaluated at the track level by aggregating segment probabilities.

    Uses the same fixed track split as the baseline.
    """

    def __init__(self, df, train_tracks, test_tracks,
                 label_col=TARGET_COL, group_col=GROUP_COL,
                 poly_degree=2):
        self.df = df.copy()
        self.label_col = label_col
        self.group_col = group_col
        self.train_tracks = np.array(train_tracks)
        self.test_tracks = np.array(test_tracks)
        self.poly_degree = poly_degree

        # Subset segments by track split
        self.df_train = self.df[self.df[group_col].isin(self.train_tracks)]
        self.df_test = self.df[self.df[group_col].isin(self.test_tracks)]

        # Numeric features only
        self.X_train_num = self.df_train.select_dtypes(include=['float64', 'int64'])
        self.X_test_num = self.df_test.select_dtypes(include=['float64', 'int64'])

        # Segment-level labels
        self.y_train = self.df_train[label_col]
        self.y_test = self.df_test[label_col]

        # Will store transformed matrices
        self.X_train = None
        self.X_test = None

        # keep transformers so they can be reused consistently
        self.poly = None
        self.scaler = None

        self.model = None

    def feature_engineering(self):
        """
        Apply polynomial feature expansion + MinMax scaling.
        """
        self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        X_train_poly = self.poly.fit_transform(self.X_train_num)
        X_test_poly = self.poly.transform(self.X_test_num)

        self.scaler = MinMaxScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_poly)
        X_test_scaled = self.scaler.transform(X_test_poly)

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled

        return self

    def fit(self, path=None):
        """
        Train (or load) logistic regression on engineered features
        and evaluate at track level.
        """
        if self.X_train is None or self.X_test is None:
            self.feature_engineering()

        if path is None:
            log_reg = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=2000,
                n_jobs=-1
            )
            log_reg.fit(self.X_train, self.y_train)
            dump(log_reg, "logistic_regression_fe.joblib")
            self.model = log_reg
        else:
            self.model = load(path)

        # Predict segment-level probabilities on test set
        proba_segments = self.model.predict_proba(self.X_test)
        classes = self.model.classes_

        proba_df = pd.DataFrame(
            proba_segments,
            index=self.df_test.index,
            columns=classes
        )
        proba_df[self.group_col] = self.df_test[self.group_col].values

        # Aggregate segment probabilities to track level (mean)
        track_proba = proba_df.groupby(self.group_col)[classes].mean()
        y_pred = track_proba.idxmax(axis=1)

        # True track-level labels
        y_true = track_proba.index.map(
            lambda g: self.df.loc[self.df[self.group_col] == g, self.label_col].iloc[0]
        )

        overall_acc = accuracy_score(y_true, y_pred)

        # Per-class accuracy
        per_class_accuracy = {}
        for c in classes:
            mask = (y_true == c)
            total = mask.sum()
            correct = (y_pred[mask] == c).sum() if total > 0 else 0
            per_class_accuracy[c] = correct / total if total > 0 else np.nan

        per_class_df = pd.DataFrame.from_dict(
            per_class_accuracy,
            orient='index',
            columns=['accuracy']
        )

        return {
            "overall accuracy": overall_acc,
            "per class accuracy": per_class_df
        }
    
# DL model
class TrackLevelDeepModel:

    def __init__(self, num_classes=None, input_mode="csv"):
        """
        input_mode: "csv", "png", or "both"
        This flag controls only how you *prepare* X_train_seq / X_test_seq
        before passing them here. The model itself always sees sequences
        shaped as (T_i, F).
        """
        self.num_classes = num_classes
        self.model = None
        self.max_len = None
        self.n_features = None
        self.scaler = None
        self.input_mode = input_mode  # just stored for clarity/debugging

    def lstm_encoder(self, x):
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        return x

    def attention_block(self, lstm_output):
        score = Dense(1)(lstm_output)
        score = Lambda(lambda x: tf.nn.softmax(x, axis=1))(score)
        context = Multiply()([lstm_output, score])
        context = Lambda(lambda x: K.sum(x, axis=1))(context)
        return context

    def cnn_encoder(self, x):
        x = Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
        x = Conv1D(64, kernel_size=1, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = GlobalAveragePooling1D()(x)
        return x

    def build_model(self, input_shape):
        """
        input_shape = (T, F) for the padded sequences.

        This same architecture works for:
        - CSV features (tabular sequences)
        - PNG-derived mel features (time x n_mels)
        - Concatenated CSV+PNG features (if you prepare them that way)
        """
        n_features = input_shape[1]
        inputs = Input(shape=(None, n_features))

        x = Masking(mask_value=0.0)(inputs)

        lstm_out = self.lstm_encoder(x)
        att_vec = self.attention_block(lstm_out)
        cnn_vec = self.cnn_encoder(lstm_out)

        merged = Concatenate()([att_vec, cnn_vec])
        merged = Dense(64, activation='relu')(merged)
        merged = Dropout(0.3)(merged)

        outputs = Dense(self.num_classes, activation='softmax')(merged)

        self.model = Model(inputs, outputs)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return self.model

    def fit(self, X_train_seq, y_train, X_test_seq, y_test, batch=32, epochs=20):
        """
        Fit the model on variable-length sequences for CSV, PNG, or combined.

        X_train_seq, X_test_seq: list of np.ndarray with shape (T_i, F)
        y_train, y_test: one-hot labels
        """
        if self.num_classes is None:
            self.num_classes = y_train.shape[1]

        # max sequence length across train+test
        self.max_len = max(len(s) for s in X_train_seq + X_test_seq)
        self.n_features = X_train_seq[0].shape[1]

        # --- feature scaling across time for each feature ---
        X_train_concat = np.concatenate(X_train_seq, axis=0)  # (sum_T, F)
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_train_concat)

        def scale_sequence_list(seq_list):
            return [self.scaler.transform(s) for s in seq_list]

        X_train_scaled = scale_sequence_list(X_train_seq)
        X_test_scaled = scale_sequence_list(X_test_seq)

        # --- pad sequences with zeros (works for CSV/PNG/both) ---
        X_train_pad = pad_sequences(
            X_train_scaled, maxlen=self.max_len,
            padding='post', dtype='float32', value=0.0
        )
        X_test_pad = pad_sequences(
            X_test_scaled, maxlen=self.max_len,
            padding='post', dtype='float32', value=0.0
        )

        # build and train model
        self.build_model((self.max_len, self.n_features))

        history = self.model.fit(
            X_train_pad, y_train,
            validation_data=(X_test_pad, y_test),
            epochs=epochs,
            batch_size=batch
        )

        return history

    def save(self, prefix="deep_track_model"):
        self.model.save(f"{prefix}.h5")
        np.save(f"{prefix}_max_len.npy", np.array([self.max_len]))
        np.save(f"{prefix}_n_features.npy", np.array([self.n_features]))

    def load(self, prefix="deep_track_model"):
        self.model = tf.keras.models.load_model(
            f"{prefix}.h5",
            custom_objects={"K": K}
        )
        self.max_len = int(np.load(f"{prefix}_max_len.npy")[0])
        self.n_features = int(np.load(f"{prefix}_n_features.npy")[0])

    def predict_sequences(self, X_seq):
        """
        X_seq: list of arrays of shape (T_i, n_features)
        Returns: softmax probability array (num_samples, num_classes)

        IMPORTANT:
        - X_seq must already be scaled with self.scaler and
          padded the same way as in fit, OR you call this after
          reusing the same scaling+padding logic externally.
        """
        # scale using fitted scaler
        X_scaled = [self.scaler.transform(s) for s in X_seq]
        X_pad = pad_sequences(
            X_scaled, maxlen=self.max_len,
            padding="post", dtype="float32", value=0.0
        )
        return self.model.predict(X_pad)

    def predict_tracks(self, df_subset, X_seq, groups_col=GROUP_COL):
        """
        df_subset: one row per track, aligned with X_seq
        X_seq: list of segment sequences (same order as df_subset)
        groups_col: track identifier

        Works for CSV, PNG, or combined inputs (as long as X_seq is (T_i, F)).
        """
        proba = self.predict_sequences(X_seq)  # [num_tracks, num_classes]

        dfp = pd.DataFrame(
            proba,
            columns=[f"class_{i}" for i in range(self.num_classes)]
        )
        dfp[groups_col] = df_subset[groups_col].values

        track_proba = dfp.groupby(groups_col).mean()
        track_pred_idx = track_proba.values.argmax(axis=1)
        track_pred = pd.Series(track_pred_idx, index=track_proba.index)

        return track_proba, track_pred

    def evaluate_tracks(self, df_subset, X_seq, y_true_track, groups_col=GROUP_COL):
        """
        Track-level evaluation for CSV, PNG, or combined sequences.
        y_true_track: integer-encoded labels (0..num_classes-1)
        """
        track_proba, track_pred = self.predict_tracks(
            df_subset=df_subset,
            X_seq=X_seq,
            groups_col=groups_col
        )

        filenames = df_subset[groups_col].values
        y_true_series = pd.Series(y_true_track, index=filenames)

        overall_acc = accuracy_score(y_true_series, track_pred)

        per_class_accuracy = {}
        for c in range(self.num_classes):
            mask = (y_true_series == c)
            total = mask.sum()
            correct = (track_pred[mask] == c).sum() if total > 0 else 0
            per_class_accuracy[c] = correct / total if total > 0 else None

        per_class_df = pd.DataFrame.from_dict(
            per_class_accuracy,
            orient='index',
            columns=['accuracy']
        )

        return {
            'overall accuracy': overall_acc,
            'per class accuracy': per_class_df
        }
    
# Main
feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

X_seq_csv, y_seq_csv, df_tracks_csv = build_track_sequences(
    df=df,
    feature_cols=feature_cols,
    label_col='label_seg',
    group_col='filename'
)

track_names_csv = df_tracks_csv[GROUP_COL].to_numpy()

# 2. Track-level split (unchanged)
(
    X_train_seq_csv,
    X_test_seq_csv,
    y_train_track_raw_csv,
    y_test_track_raw_csv,
    groups_train_csv,
    groups_test_csv
) = split_sequences(
    groups=track_names_csv,
    X_seq=X_seq_csv,
    y_track=y_seq_csv,
    test_size=0.2,
    random_state=12345
)

# 3. Encode labels for deep model (CSV)
le_csv = LabelEncoder()
y_train_enc_csv = le_csv.fit_transform(y_train_track_raw_csv)
y_test_enc_csv = le_csv.transform(y_test_track_raw_csv)

num_classes_csv = len(le_csv.classes_)

y_train_cat_csv = to_categorical(y_train_enc_csv, num_classes=num_classes_csv)
y_test_cat_csv = to_categorical(y_test_enc_csv, num_classes=num_classes_csv)

# 4. Build df_train_tracks / df_test_tracks for evaluation (CSV)
df_train_tracks_csv = df_tracks_csv[df_tracks_csv[GROUP_COL].isin(groups_train_csv)].reset_index(drop=True)
df_test_tracks_csv = df_tracks_csv[df_tracks_csv[GROUP_COL].isin(groups_test_csv)].reset_index(drop=True)

# --------------------------------------------------------
# 6.4 Baseline Logistic Regression (segment-level, aggregated)
#     using the same track split
# --------------------------------------------------------
# print("\n=== Baseline Logistic Regression (segment-level, aggregated) ===")
# logistic_baseline = LogisticSegmentModelWithFixedSplit(
#     df=df,
#     train_tracks=groups_train,
#     test_tracks=groups_test,
#     label_col=TARGET_COL,
#     group_col=GROUP_COL
# )
# baseline_results = logistic_baseline.fit()
# print("Overall accuracy:", baseline_results["overall accuracy"])
# print("Per-class accuracy:\n", baseline_results["per class accuracy"])

# # --------------------------------------------------------
# # 6.5 Logistic Regression with Feature Engineering (segment-level, aggregated)
# #     using the same track split
# # --------------------------------------------------------
# print("\n=== Logistic Regression with Feature Engineering (segment-level, aggregated) ===")
# logistic_fe = LogisticSegmentModelWithFE(
#     df=df,
#     train_tracks=groups_train,
#     test_tracks=groups_test,
#     label_col=TARGET_COL,
#     group_col=GROUP_COL,
#     poly_degree=2
# )
# fe_results = logistic_fe.fit()
# print("Overall accuracy:", fe_results["overall accuracy"])
# print("Per-class accuracy:\n", fe_results["per class accuracy"])

# 5. Deep Learning Model on CSV only
print("\n=== Deep Learning Model (track-level, CSV sequence-based) ===")
deep_model_csv = TrackLevelDeepModel(num_classes=num_classes_csv, input_mode="csv")

history_csv = deep_model_csv.fit(
    X_train_seq=X_train_seq_csv,
    y_train=y_train_cat_csv,
    X_test_seq=X_test_seq_csv,
    y_test=y_test_cat_csv,
    epochs=20
)

# Optionally save CSV model separately
deep_model_csv.save("track_classifier_csv")

# Evaluate on test set (track-level, CSV)
deep_results_csv = deep_model_csv.evaluate_tracks(
    df_subset=df_test_tracks_csv,
    X_seq=X_test_seq_csv,
    y_true_track=y_test_enc_csv,
    groups_col=GROUP_COL
)
print("CSV Overall accuracy:", deep_results_csv['overall accuracy'])
print("CSV Per-class accuracy:\n", deep_results_csv['per class accuracy'])

import os
import re
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# 1. Configuration
# ============================================================

IMAGES_ROOT = r"C:\Users\apass\Downloads\gtzan\Data\images_original"

# We will reuse these names so they plug into your existing DL code:
TARGET_COL = 'label_seg'   # will hold genre label (for consistency)
GROUP_COL = 'filename'     # track identifier, e.g. "blues.00000.wav"

# ============================================================
# 2. Helper to derive track "filename" from image path
# ============================================================

def image_path_to_filename_and_label(img_path):
    """
    For a path like:
        images_original/blues/blues00000.png
    We extract:
        genre   = 'blues'   (the parent folder)
        base    = 'blues00000'
        filename= 'blues00000.wav'  (no dot, matches the base)
    """
    genre = os.path.basename(os.path.dirname(img_path))  # parent folder
    base = os.path.splitext(os.path.basename(img_path))[0]
    filename = base + ".wav"
    return filename, genre


# ============================================================
# 3. Load mel spectrogram images into a "segment-level" dataframe
# ============================================================

def load_mel_images_as_segments(images_root=IMAGES_ROOT,
                                group_col=GROUP_COL,
                                label_col=TARGET_COL):
    print("=== Debug: load_mel_images_as_segments ===")
    print("Current working directory:", os.getcwd())
    print("Absolute images_root     :", os.path.abspath(images_root))

    pattern_png = os.path.join(images_root, '**', '*.png')
    pattern_jpg = os.path.join(images_root, '**', '*.jpg')
    pattern_jpeg = os.path.join(images_root, '**', '*.jpeg')

    png_paths = glob(pattern_png, recursive=True)
    jpg_paths = glob(pattern_jpg, recursive=True)
    jpeg_paths = glob(pattern_jpeg, recursive=True)

    if png_paths:
        all_paths = png_paths
        print("Using PNG files.")
    elif jpg_paths:
        all_paths = jpg_paths
        print("Using JPG files.")
    elif jpeg_paths:
        all_paths = jpeg_paths
        print("Using JPEG files.")
    else:
        print("WARNING: No .png/.jpg/.jpeg files found under:", images_root)
        return pd.DataFrame(), []

    print("Example paths:", all_paths[:5])

    records = []
    X_img = []

    for p in all_paths:
        filename, genre = image_path_to_filename_and_label(p)

        img = Image.open(p).convert('L')  # grayscale
        arr = np.array(img, dtype=np.float32)

        # scale pixels to [0, 1]
        arr = arr / 255.0

        records.append({group_col: filename, label_col: genre})
        X_img.append(arr)

    df_mel = pd.DataFrame(records)

    # Normalize mel filenames to match tabular naming, e.g.
    # blues00000.wav -> blues.00000.wav
    df_mel[group_col] = df_mel[group_col].str.replace(
        r'(\D+)(\d{5})\.wav$',
        r'\1.\2.wav',
        regex=True
    )


    print("Loaded mel segments:", len(df_mel))
    print("df_mel columns:", df_mel.columns.tolist())
    print(df_mel.head())

    return df_mel, X_img


# ============================================================
# 4. Build per-track sequences from mel images
# ============================================================

def build_mel_track_sequences(df_mel, X_img,
                              label_col=TARGET_COL,
                              group_col=GROUP_COL):
    """
    Build per-track sequences from mel spectrogram images.

    Each image arr: (H, W) -> sequence (time_frames=W, n_mels=H) via arr.T.
    For each track, all its images are concatenated along time.
    """
    df_mel = df_mel.copy()
    df_mel['img_idx'] = np.arange(len(df_mel))

    track_names = df_mel[group_col].unique()

    X_seq_mel = []
    y_seq_mel = []
    df_tracks_rows = []

    for fname in track_names:
        dfg = df_mel[df_mel[group_col] == fname].copy()
        dfg = dfg.sort_values('img_idx')

        track_segments = []

        for idx in dfg['img_idx']:
            arr = X_img[idx].astype('float32')  # (H, W)
            if arr.max() > 1.0:
                arr = arr / 255.0

            arr_seq = arr.T  # (time_frames, n_mels)
            track_segments.append(arr_seq)

        track_seq = np.concatenate(track_segments, axis=0)  # (T_total, n_mels)
        X_seq_mel.append(track_seq)

        y_seq_mel.append(dfg[label_col].iloc[0])
        df_tracks_rows.append(dfg.iloc[0])

    df_tracks_mel = pd.DataFrame(df_tracks_rows)

    return X_seq_mel, np.array(y_seq_mel), df_tracks_mel


# ============================================================
# 5. Track-level split for mel sequences (same logic as before)
# ============================================================

def split_mel_sequences(df_tracks_mel, X_seq_mel, y_seq_mel,
                        test_size=0.2, random_state=12345,
                        group_col=GROUP_COL):
    """
    Track-level split for mel-based sequences.
    """
    track_names = df_tracks_mel[group_col].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_test = next(gss.split(X_seq_mel, y=y_seq_mel, groups=track_names))

    X_train_seq_mel = [X_seq_mel[i] for i in idx_train]
    X_test_seq_mel = [X_seq_mel[i] for i in idx_test]

    y_train_mel_raw = y_seq_mel[idx_train]
    y_test_mel_raw = y_seq_mel[idx_test]

    groups_train_mel = track_names[idx_train]
    groups_test_mel = track_names[idx_test]

    return (
        X_train_seq_mel,
        X_test_seq_mel,
        y_train_mel_raw,
        y_test_mel_raw,
        groups_train_mel,
        groups_test_mel
    )


# ============================================================
# 6. Label encoding and padding for the existing DL model
# ============================================================

def prepare_mel_for_deep_model(X_train_seq_mel, X_test_seq_mel,
                               y_train_mel_raw, y_test_mel_raw):
    """
    Encode labels and prepare them for TrackLevelDeepModel (mel only).
    """
    le_mel = LabelEncoder()
    y_train_enc_mel = le_mel.fit_transform(y_train_mel_raw)
    y_test_enc_mel = le_mel.transform(y_test_mel_raw)

    num_classes_mel = len(le_mel.classes_)
    y_train_cat_mel = to_categorical(y_train_enc_mel, num_classes=num_classes_mel)
    y_test_cat_mel = to_categorical(y_test_enc_mel, num_classes=num_classes_mel)

    return (
        X_train_seq_mel,
        X_test_seq_mel,
        y_train_cat_mel,
        y_test_cat_mel,
        num_classes_mel,
        le_mel,
        y_train_enc_mel,
        y_test_enc_mel
    )

# ============================================================
# 7. Example usage (can be integrated into your main script)
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 0. Prepare tabular data (CSV features)
    # ============================================================
    # feature columns: all numeric
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # build per-track sequences for tabular data
    X_seq_csv, y_seq_csv, df_tracks_csv = build_track_sequences(
        df=df,
        feature_cols=feature_cols,
        label_col=TARGET_COL,
        group_col=GROUP_COL
    )

    track_names_csv = df_tracks_csv[GROUP_COL].to_numpy()

    # track-level split (same groups used for all tabular tests)
    (
        X_train_seq_csv,
        X_test_seq_csv,
        y_train_track_raw_csv,
        y_test_track_raw_csv,
        groups_train_csv,
        groups_test_csv
    ) = split_sequences(
        groups=track_names_csv,
        X_seq=X_seq_csv,
        y_track=y_seq_csv,
        test_size=0.2,
        random_state=12345
    )

    # encode labels for deep model (tabular)
    le_csv = LabelEncoder()
    y_train_enc_csv = le_csv.fit_transform(y_train_track_raw_csv)
    y_test_enc_csv = le_csv.transform(y_test_track_raw_csv)

    num_classes_csv = len(le_csv.classes_)
    y_train_cat_csv = to_categorical(y_train_enc_csv, num_classes=num_classes_csv)
    y_test_cat_csv = to_categorical(y_test_enc_csv, num_classes=num_classes_csv)

    # track-level metadata for evaluation
    df_train_tracks_csv = df_tracks_csv[df_tracks_csv[GROUP_COL].isin(groups_train_csv)].reset_index(drop=True)
    df_test_tracks_csv = df_tracks_csv[df_tracks_csv[GROUP_COL].isin(groups_test_csv)].reset_index(drop=True)

    # ============================================================
    # TEST 1: Baseline Logistic Regression (tabular, segment-level aggregated)
    # ============================================================
    print("\n=== TEST 1: Baseline Logistic Regression (tabular) ===")
    logistic_baseline = LogisticSegmentModelWithFixedSplit(
        df=df,
        train_tracks=groups_train_csv,
        test_tracks=groups_test_csv,
        label_col=TARGET_COL,
        group_col=GROUP_COL
    )
    baseline_results = logistic_baseline.fit()
    print("Baseline Overall accuracy:", baseline_results["overall accuracy"])
    print("Baseline Per-class accuracy:\n", baseline_results["per class accuracy"])

    # ============================================================
    # TEST 2: Logistic Regression + Feature Engineering (tabular)
    # ============================================================
    print("\n=== TEST 2: Logistic Regression with Feature Engineering (tabular) ===")
    logistic_fe = LogisticSegmentModelWithFE(
        df=df,
        train_tracks=groups_train_csv,
        test_tracks=groups_test_csv,
        label_col=TARGET_COL,
        group_col=GROUP_COL,
        poly_degree=2
    )
    fe_results = logistic_fe.fit()
    print("FE Overall accuracy:", fe_results["overall accuracy"])
    print("FE Per-class accuracy:\n", fe_results["per class accuracy"])

    # ============================================================
    # TEST 3: Deep Learning (tabular, sequence-based)
    # ============================================================
    print("\n=== TEST 3: Deep Learning Model (track-level, TABULAR sequence-based) ===")
    deep_model_tab = TrackLevelDeepModel(num_classes=num_classes_csv, input_mode="csv")
    history_tab = deep_model_tab.fit(
        X_train_seq=X_train_seq_csv,
        y_train=y_train_cat_csv,
        X_test_seq=X_test_seq_csv,
        y_test=y_test_cat_csv,
        epochs=20,
        batch=32
    )

    deep_model_tab.save("track_classifier_tabular")

    deep_results_tab = deep_model_tab.evaluate_tracks(
        df_subset=df_test_tracks_csv,
        X_seq=X_test_seq_csv,
        y_true_track=y_test_enc_csv,
        groups_col=GROUP_COL
    )
    print("TABULAR Overall accuracy:", deep_results_tab['overall accuracy'])
    print("TABULAR Per-class accuracy:\n", deep_results_tab['per class accuracy'])

    # ============================================================
    # 4. Prepare MEL spectrogram data
    # ============================================================
    print("\n=== Loading MEL spectrogram images ===")
    df_mel, X_img = load_mel_images_as_segments(IMAGES_ROOT)

    if df_mel.empty:
        raise RuntimeError(
            "No mel images loaded (df_mel is empty). "
            "Check IMAGES_ROOT, folder structure, and file extensions."
        )

    X_seq_mel, y_seq_mel, df_tracks_mel = build_mel_track_sequences(
        df_mel=df_mel,
        X_img=X_img,
        label_col=TARGET_COL,
        group_col=GROUP_COL
    )

    print("Number of mel tracks:", len(X_seq_mel))
    print("Example MEL sequence shape (T_i, F_mel):", X_seq_mel[0].shape)

    (
        X_train_seq_mel,
        X_test_seq_mel,
        y_train_mel_raw,
        y_test_mel_raw,
        groups_train_mel,
        groups_test_mel
    ) = split_mel_sequences(
        df_tracks_mel=df_tracks_mel,
        X_seq_mel=X_seq_mel,
        y_seq_mel=y_seq_mel,
        test_size=0.2,
        random_state=12345,
        group_col=GROUP_COL
    )

    (
        X_train_seq_mel,
        X_test_seq_mel,
        y_train_cat_mel,
        y_test_cat_mel,
        num_classes_mel,
        le_mel,
        y_train_enc_mel,
        y_test_enc_mel
    ) = prepare_mel_for_deep_model(
        X_train_seq_mel=X_train_seq_mel,
        X_test_seq_mel=X_test_seq_mel,
        y_train_mel_raw=y_train_mel_raw,
        y_test_mel_raw=y_test_mel_raw
    )

    df_test_tracks_mel = df_tracks_mel[df_tracks_mel[GROUP_COL].isin(groups_test_mel)].reset_index(drop=True)

    # ============================================================
    # TEST 4: Deep Learning (MEL spectrogram sequences)
    # ============================================================
    print("\n=== TEST 4: Deep Learning Model (track-level, MEL sequence-based) ===")
    deep_model_mel = TrackLevelDeepModel(num_classes=num_classes_mel, input_mode="png")
    history_mel = deep_model_mel.fit(
        X_train_seq=X_train_seq_mel,
        y_train=y_train_cat_mel,
        X_test_seq=X_test_seq_mel,
        y_test=y_test_cat_mel,
        epochs=20,
        batch=32
    )

    deep_model_mel.save("track_classifier_mel")

    deep_results_mel = deep_model_mel.evaluate_tracks(
        df_subset=df_test_tracks_mel,
        X_seq=X_test_seq_mel,
        y_true_track=y_test_enc_mel,
        groups_col=GROUP_COL
    )
    print("MEL Overall accuracy:", deep_results_mel['overall accuracy'])
    print("MEL Per-class accuracy:\n", deep_results_mel['per class accuracy'])

    # ============================================================
    # TEST 5: Deep Learning (TABULAR + MEL concatenated sequences)
    # ============================================================
    print("\n=== TEST 5: Deep Learning Model (TABULAR + MEL concatenated sequences) ===")

    # 1) Find tracks that exist in both tabular and mel sets
    csv_fnames = df_tracks_csv[GROUP_COL].unique()
    mel_fnames = df_tracks_mel[GROUP_COL].unique()
    common_tracks = np.intersect1d(csv_fnames, mel_fnames)

    if len(common_tracks) == 0:
        raise RuntimeError("No common tracks between tabular and mel data for combination test.")

    # 2) Build dicts: filename -> sequence
    csv_seq_dict = {fname: seq for fname, seq in zip(df_tracks_csv[GROUP_COL], X_seq_csv)}
    mel_seq_dict = {fname: seq for fname, seq in zip(df_tracks_mel[GROUP_COL], X_seq_mel)}

    # labels from tabular side (string labels)
    label_dict = {fname: label for fname, label in zip(df_tracks_csv[GROUP_COL], y_seq_csv)}

    X_seq_combined = []
    y_seq_combined = []
    filenames_combined = []

    for fname in common_tracks:
        seq_csv = csv_seq_dict[fname]  # (T_csv, F_csv)
        seq_mel = mel_seq_dict[fname]  # (T_mel, F_mel)

        # Make sequence lengths match by truncating to the minimum T
        T_min = min(seq_csv.shape[0], seq_mel.shape[0])
        seq_csv_trim = seq_csv[:T_min, :]
        seq_mel_trim = seq_mel[:T_min, :]

        # Concatenate along feature dimension
        seq_comb = np.concatenate([seq_csv_trim, seq_mel_trim], axis=1)  # (T_min, F_csv + F_mel)

        X_seq_combined.append(seq_comb)
        y_seq_combined.append(label_dict[fname])
        filenames_combined.append(fname)

    X_seq_combined = list(X_seq_combined)
    y_seq_combined = np.array(y_seq_combined)

    # 3) Build a simple df_tracks_combined with a clear filename column
    df_tracks_combined = pd.DataFrame({
        GROUP_COL: filenames_combined,
        TARGET_COL: y_seq_combined
    })

    # 4) Split combined sequences at track level
    (
        X_train_seq_comb,
        X_test_seq_comb,
        y_train_comb_raw,
        y_test_comb_raw,
        groups_train_comb,
        groups_test_comb
    ) = split_sequences(
        groups=df_tracks_combined[GROUP_COL].to_numpy(),
        X_seq=X_seq_combined,
        y_track=y_seq_combined,
        test_size=0.2,
        random_state=12345
    )

    # 5) Encode labels and prepare for deep model
    le_comb = LabelEncoder()
    y_train_enc_comb = le_comb.fit_transform(y_train_comb_raw)
    y_test_enc_comb = le_comb.transform(y_test_comb_raw)

    num_classes_comb = len(le_comb.classes_)
    y_train_cat_comb = to_categorical(y_train_enc_comb, num_classes=num_classes_comb)
    y_test_cat_comb = to_categorical(y_test_enc_comb, num_classes=num_classes_comb)

    df_test_tracks_comb = df_tracks_combined[df_tracks_combined[GROUP_COL].isin(groups_test_comb)].reset_index(drop=True)

    # 6) Train combined deep model
    deep_model_comb = TrackLevelDeepModel(num_classes=num_classes_comb, input_mode="both")
    history_comb = deep_model_comb.fit(
        X_train_seq=X_train_seq_comb,
        y_train=y_train_cat_comb,
        X_test_seq=X_test_seq_comb,
        y_test=y_test_cat_comb,
        epochs=20,
        batch=32
    )

    deep_model_comb.save("track_classifier_tabular_mel")

    deep_results_comb = deep_model_comb.evaluate_tracks(
        df_subset=df_test_tracks_comb,
        X_seq=X_test_seq_comb,
        y_true_track=y_test_enc_comb,
        groups_col=GROUP_COL
    )
    print("TABULAR+MEL Overall accuracy:", deep_results_comb['overall accuracy'])
    print("TABULAR+MEL Per-class accuracy:\n", deep_results_comb['per class accuracy'])