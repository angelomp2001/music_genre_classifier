# # libraries
# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, LabelEncoder
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from joblib import dump, load

# import tensorflow as tf
# from tensorflow.keras.layers import (
#     Input, LSTM, Bidirectional, Dense, Dropout, Conv1D,
#     GlobalAveragePooling1D, Multiply, Lambda, Concatenate, Activation, Masking)

# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# import tensorflow.keras.backend as K


# # EDA
# features_3_sec_path = 'Data/features_3_sec.csv' # C:\Users\apass\OneDrive\Documents\VS code\Music_genre_predictor\Data
# features_30_sec_path = 'Data/features_30_sec.csv' # C:\Users\apass\OneDrive\Documents\VS code\Music_genre_predictor\Data
# df = pd.read_csv(features_3_sec_path) # C:\Users\Angelo\Downloads\GTZAN data set\Data
# # df.shape # (9990, 60)
# # df.dtypes # filename, label = object. length = int. the rest = float
# # df.isna().sum().sum() # np.int64(0)
# df_30_sec = pd.read_csv(features_30_sec_path) # C:\Users\Angelo\Downloads\GTZAN data set\Data
# df_30_sec.head()
# df.head()


# # merge dfs
# # create index to match index of other df, save as 'common_key'
# df = df.copy()
# df.rename(columns={'filename': 'file_segment'}, inplace = True)
# df["filename"] = (
#     df['file_segment']
#     .str.replace(r"\.\d+\.wav$", ".wav", regex=True)
# )

# # move it to loc 0, so I can see it.
# df.insert(0, 'filename', df.pop('filename'))

# df = (
#     df_30_sec
#     .merge(df, how = 'left', on = 'filename', suffixes = ('_file', '_seg'))
# )

# df.insert(1, 'file_segment', df.pop('file_segment'))

# TARGET_COL = 'label_seg'
# GROUP_COL = 'filename'


# # split data function
# def build_track_sequences(df, feature_cols, label_col=TARGET_COL, group_col=GROUP_COL):
#     """
#     Build per-track sequences from a segment-level dataframe.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Must contain at least [group_col, label_col] and feature_cols.
#     feature_cols : list of str
#         Names of numeric feature columns to use as segment features.
#     label_col : str
#         Segment-level label column; assumed constant within each track.
#     group_col : str
#         Column representing the track identifier (e.g., filename).

#     Returns
#     -------
#     X_seq : list of np.ndarray
#         Each element has shape (T_i, n_features) for track i.
#     y_track : np.ndarray
#         Array of track-level labels, one per track.
#     df_subset : pd.DataFrame
#         One row per track (metadata).
#     """
#     groups = df[group_col].unique()

#     X_seq = []
#     y_track = []
#     df_subset_rows = []

#     for g in groups:
#         dfg = df[df[group_col] == g]
#         # Each row is one segment; use all numeric features in feature_cols
#         seq = dfg[feature_cols].to_numpy()  # shape: (T_i, F)
#         X_seq.append(seq)

#         # Track-level label: assume all segments share same label
#         y_track.append(dfg[label_col].iloc[0])

#         # Keep one representative row for metadata
#         df_subset_rows.append(dfg.iloc[0])

#     df_subset = pd.DataFrame(df_subset_rows)

#     return X_seq, np.array(y_track), df_subset

# # split sequences
# def split_sequences(groups, X_seq, y_track, test_size=0.2, random_state=12345):
#     """
#     Track-level split for sequences.

#     Parameters
#     ----------
#     groups : array-like
#         Array of track identifiers, one per sequence.
#     X_seq : list of np.ndarray
#         List of sequences, length = num_tracks.
#     y_track : np.ndarray
#         Track-level labels, length = num_tracks.
#     test_size : float
#         Fraction of tracks to use for test.
#     random_state : int
#         Random seed.

#     Returns
#     -------
#     X_train_seq, X_test_seq, y_train, y_test, groups_train, groups_test
#     """
#     gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
#     idx_train, idx_test = next(gss.split(X_seq, y=y_track, groups=groups))

#     X_train_seq = [X_seq[i] for i in idx_train]
#     X_test_seq = [X_seq[i] for i in idx_test]

#     y_train = y_track[idx_train]
#     y_test = y_track[idx_test]

#     groups = np.asarray(groups)
#     groups_train = groups[idx_train]
#     groups_test = groups[idx_test]

#     return X_train_seq, X_test_seq, y_train, y_test, groups_train, groups_test

# # logistic regression classes
# class LogisticSegmentModelWithFixedSplit:
#     """
#     Logistic regression trained at the segment level (3-sec segments),
#     evaluated at the track level by aggregating segment probabilities.

#     Uses a fixed track-level split: train_tracks vs test_tracks (same as deep model).
#     """

#     def __init__(self, df, train_tracks, test_tracks,
#                  label_col=TARGET_COL, group_col=GROUP_COL):
#         self.df = df.copy()
#         self.label_col = label_col
#         self.group_col = group_col
#         self.train_tracks = np.array(train_tracks)
#         self.test_tracks = np.array(test_tracks)

#         # Subset segments by track split
#         self.df_train = self.df[self.df[group_col].isin(self.train_tracks)]
#         self.df_test = self.df[self.df[group_col].isin(self.test_tracks)]

#         # Numeric features only
#         self.X_train = self.df_train.select_dtypes(include=['float64', 'int64'])
#         self.X_test = self.df_test.select_dtypes(include=['float64', 'int64'])

#         # Segment-level labels
#         self.y_train = self.df_train[label_col]
#         self.y_test = self.df_test[label_col]

#         self.model = None

#     def fit(self, path=None):
#         """
#         Train (or load) logistic regression and evaluate at track level.
#         Returns a dictionary with overall and per-class accuracy.
#         """
#         if path is None:
#             log_reg = LogisticRegression(
#                 multi_class="multinomial",
#                 solver="lbfgs",
#                 max_iter=2000,
#                 n_jobs=-1
#             )
#             log_reg.fit(self.X_train, self.y_train)
#             dump(log_reg, "logistic_regression_baseline.joblib")
#             self.model = log_reg
#         else:
#             self.model = load(path)

#         # Predict segment-level probabilities on test segments
#         proba_segments = self.model.predict_proba(self.X_test)
#         classes = self.model.classes_

#         proba_df = pd.DataFrame(
#             proba_segments,
#             index=self.df_test.index,
#             columns=classes
#         )
#         proba_df[self.group_col] = self.df_test[self.group_col].values

#         # Aggregate segment probabilities to track level (mean)
#         track_proba = proba_df.groupby(self.group_col)[classes].mean()
#         y_pred = track_proba.idxmax(axis=1)

#         # True track-level labels
#         y_true = track_proba.index.map(
#             lambda g: self.df.loc[self.df[self.group_col] == g, self.label_col].iloc[0]
#         )

#         overall_acc = accuracy_score(y_true, y_pred)

#         # Per-class accuracy
#         per_class_accuracy = {}
#         for c in classes:
#             mask = (y_true == c)
#             total = mask.sum()
#             correct = (y_pred[mask] == c).sum() if total > 0 else 0
#             per_class_accuracy[c] = correct / total if total > 0 else np.nan

#         per_class_df = pd.DataFrame.from_dict(
#             per_class_accuracy,
#             orient='index',
#             columns=['accuracy']
#         )

#         return {
#             "overall accuracy": overall_acc,
#             "per class accuracy": per_class_df
#         }

# # ============================================================
# # 4. Segment-Level Logistic Regression with Feature Engineering
# #    (Track Aggregation)
# # ============================================================

# class LogisticSegmentModelWithFE:
#     """
#     Logistic regression with polynomial feature engineering at the segment level,
#     evaluated at the track level by aggregating segment probabilities.

#     Uses the same fixed track split as the baseline.
#     """

#     def __init__(self, df, train_tracks, test_tracks,
#                  label_col=TARGET_COL, group_col=GROUP_COL,
#                  poly_degree=2):
#         self.df = df.copy()
#         self.label_col = label_col
#         self.group_col = group_col
#         self.train_tracks = np.array(train_tracks)
#         self.test_tracks = np.array(test_tracks)
#         self.poly_degree = poly_degree

#         # Subset segments by track split
#         self.df_train = self.df[self.df[group_col].isin(self.train_tracks)]
#         self.df_test = self.df[self.df[group_col].isin(self.test_tracks)]

#         # Numeric features only
#         self.X_train_num = self.df_train.select_dtypes(include=['float64', 'int64'])
#         self.X_test_num = self.df_test.select_dtypes(include=['float64', 'int64'])

#         # Segment-level labels
#         self.y_train = self.df_train[label_col]
#         self.y_test = self.df_test[label_col]

#         # Will store transformed matrices
#         self.X_train = None
#         self.X_test = None

#         self.model = None

#     def feature_engineering(self):
#         """
#         Apply polynomial feature expansion + MinMax scaling.
#         """
#         poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
#         X_train_poly = poly.fit_transform(self.X_train_num)
#         X_test_poly = poly.transform(self.X_test_num)

#         scaler = MinMaxScaler()
#         X_train_scaled = scaler.fit_transform(X_train_poly)
#         X_test_scaled = scaler.transform(X_test_poly)

#         self.X_train = X_train_scaled
#         self.X_test = X_test_scaled

#         return self

#     def fit(self, path=None):
#         """
#         Train (or load) logistic regression on engineered features
#         and evaluate at track level.
#         """
#         if self.X_train is None or self.X_test is None:
#             self.feature_engineering()

#         if path is None:
#             log_reg = LogisticRegression(
#                 multi_class="multinomial",
#                 solver="lbfgs",
#                 max_iter=2000,
#                 n_jobs=-1
#             )
#             log_reg.fit(self.X_train, self.y_train)
#             dump(log_reg, "logistic_regression_fe.joblib")
#             self.model = log_reg
#         else:
#             self.model = load(path)

#         # Predict segment-level probabilities on test set
#         proba_segments = self.model.predict_proba(self.X_test)
#         classes = self.model.classes_

#         proba_df = pd.DataFrame(
#             proba_segments,
#             index=self.df_test.index,
#             columns=classes
#         )
#         proba_df[self.group_col] = self.df_test[self.group_col].values

#         # Aggregate segment probabilities to track level (mean)
#         track_proba = proba_df.groupby(self.group_col)[classes].mean()
#         y_pred = track_proba.idxmax(axis=1)

#         # True track-level labels
#         y_true = track_proba.index.map(
#             lambda g: self.df.loc[self.df[self.group_col] == g, self.label_col].iloc[0]
#         )

#         overall_acc = accuracy_score(y_true, y_pred)

#         # Per-class accuracy
#         per_class_accuracy = {}
#         for c in classes:
#             mask = (y_true == c)
#             total = mask.sum()
#             correct = (y_pred[mask] == c).sum() if total > 0 else 0
#             per_class_accuracy[c] = correct / total if total > 0 else np.nan

#         per_class_df = pd.DataFrame.from_dict(
#             per_class_accuracy,
#             orient='index',
#             columns=['accuracy']
#         )

#         return {
#             "overall accuracy": overall_acc,
#             "per class accuracy": per_class_df
#         }
    
# # DL model
# class TrackLevelDeepModel:

#     def __init__(self, num_classes=None):
#         self.num_classes = num_classes
#         self.model = None
#         self.max_len = None
#         self.n_features = None

#     # 5.1 LSTM ENCODER
#     def lstm_encoder(self, x):
#         x = Bidirectional(LSTM(128, return_sequences=True))(x)
#         x = Bidirectional(LSTM(64, return_sequences=True))(x)
#         return x

#     # 5.2 ATTENTION BLOCK
#     def attention_block(self, lstm_output):
#         # # lstm_output: (batch, T, D)
#         # score = Dense(1)(lstm_output)        # (batch, T, 1)
#         # score = Activation('softmax')(score) # (batch, T, 1) over time
#         # context = Multiply()([lstm_output, score])
#         # context = Lambda(lambda x: K.sum(x, axis=1))(context)
#         # return context

#         attention = Dense(1, activation='tanh')(lstm_output)  # (batch, T, 1)
#         attention = Dense(766, activation = 'softmax')(attention)  # (batch, T, 1) over time
#         attention = Lambda(lambda x: K.expand_dims(x, axis=-1))(attention)  # (batch, T, 1, 1)
#         attention = Multiply()([lstm_output, attention])  # (batch, T, D)
#         attention = Lambda(lambda x: K.sum(x, axis=1))(attention)  # (batch, D)

#     # 5.3 CNN ENCODER
#     def cnn_encoder(self, x):
#         x = Conv1D(32, kernel_size=5, padding='same', activation='relu')(x)
#         x = Conv1D(64, kernel_size=1, activation='relu')(x)
#         x = Dropout(0.2)(x)
#         x = Conv1D(32, kernel_size=1, activation='relu')(x)
#         x = Dropout(0.2)(x)
#         x = GlobalAveragePooling1D()(x)
#         return x

#     # 5.4 BUILD MODEL
#     def build_model(self, input_shape):
#         """
#         input_shape = (T, F) for the padded sequences.
#         """
#         n_features = input_shape[1]
#         inputs = Input(shape=(None, n_features))
#         x = Masking(mask_value=0.0)(inputs)

#         lstm_out = self.lstm_encoder(inputs)
#         att_vec = self.attention_block(lstm_out)
#         cnn_vec = self.cnn_encoder(lstm_out)

#         merged = Concatenate()([att_vec, cnn_vec])
#         merged = Dense(64, activation='relu')(merged)
#         merged = Dropout(0.3)(merged)

#         outputs = Dense(self.num_classes, activation='softmax')(merged)

#         self.model = Model(inputs, outputs)
#         self.model.compile(
#             loss='categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy']
#         )
#         return self.model

#     # 5.5 FIT
#     def fit(self, X_train_seq, y_train, X_test_seq, y_test, batch=32, epochs=20):
#         if self.num_classes is None:
#             self.num_classes = y_train.shape[1]

#         # Determine max sequence length and number of features
#         self.max_len = max(len(s) for s in X_train_seq + X_test_seq)
#         self.n_features = X_train_seq[0].shape[1]

#         # Pad sequences
#         X_train_pad = pad_sequences(
#             X_train_seq, maxlen=self.max_len,
#             padding='post', dtype='float32'
#         )
#         X_test_pad = pad_sequences(
#             X_test_seq, maxlen=self.max_len,
#             padding='post', dtype='float32'
#         )

#         # Build model with fixed (max_len, n_features)
#         self.build_model((self.max_len, self.n_features))

#         history = self.model.fit(
#             X_train_pad, y_train,
#             validation_data=(X_test_pad, y_test),
#             epochs=epochs,
#             batch_size=batch
#         )

#         return history

#     # 5.6 SAVE
#     def save(self, prefix="deep_track_model"):
#         self.model.save(f"{prefix}.h5")
#         np.save(f"{prefix}_max_len.npy", np.array([self.max_len]))
#         np.save(f"{prefix}_n_features.npy", np.array([self.n_features]))

#     # 5.7 LOAD
#     def load(self, prefix="deep_track_model"):
#         self.model = tf.keras.models.load_model(
#             f"{prefix}.h5",
#             custom_objects={"K": K}
#         )
#         self.max_len = int(np.load(f"{prefix}_max_len.npy")[0])
#         self.n_features = int(np.load(f"{prefix}_n_features.npy")[0])

#     # 5.8 PREDICT SEQUENCES
#     def predict_sequences(self, X_seq):
#         """
#         X_seq: list of arrays of shape (T_i, n_features)
#         Returns: softmax probability array (num_samples, num_classes)
#         """
#         X_pad = pad_sequences(
#             X_seq, maxlen=self.max_len,
#             padding="post", dtype="float32"
#         )
#         return self.model.predict(X_pad)

#     # 5.9 TRACK-LEVEL SOFTMAX AGGREGATION
#     def predict_tracks(self, df_subset, X_seq, groups_col=GROUP_COL):
#         """
#         df_subset: dataframe aligned with X_seq construction, must contain groups_col
#         X_seq: list of segment sequences (same order as df_subset groups)
#         groups_col: column name representing track-level groups

#         Returns
#         -------
#         track_proba: DataFrame indexed by track name with average per-class probabilities
#         track_pred:  Series of predicted class indices (0..num_classes-1)
#         """
#         proba = self.predict_sequences(X_seq)  # shape: [num_tracks, num_classes]

#         dfp = pd.DataFrame(
#             proba,
#             columns=[f"class_{i}" for i in range(self.num_classes)]
#         )

#         # Assume df_subset has unique rows per track in corresponding order
#         dfp[groups_col] = df_subset[groups_col].values

#         track_proba = dfp.groupby(groups_col).mean()

#         track_pred_idx = track_proba.values.argmax(axis=1)
#         track_pred = pd.Series(track_pred_idx, index=track_proba.index)

#         return track_proba, track_pred

#     # 5.10 TRACK-LEVEL EVALUATION
#     def evaluate_tracks(self, df_subset, X_seq, y_true_track, groups_col=GROUP_COL):
#         """
#         Evaluate at track level.

#         Parameters
#         ----------
#         df_subset : pd.DataFrame
#             One row per track, aligned with X_seq.
#         X_seq : list of np.ndarray
#             One sequence per track.
#         y_true_track : array-like
#             True track-level labels (integer-encoded 0..num_classes-1).
#         groups_col : str
#             Group column identifying each track.

#         Returns
#         -------
#         dict with 'overall accuracy' and 'per class accuracy' DataFrame.
#         """
#         # Predictions
#         track_proba, track_pred = self.predict_tracks(
#             df_subset=df_subset,
#             X_seq=X_seq,
#             groups_col=groups_col
#         )

#         # y_true_track should align with df_subset and X_seq
#         filenames = df_subset[groups_col].values
#         y_true_series = pd.Series(y_true_track, index=filenames)

#         overall_acc = accuracy_score(y_true_series, track_pred)

#         # Per-class accuracy
#         per_class_accuracy = {}
#         for c in range(self.num_classes):
#             mask = (y_true_series == c)
#             total = mask.sum()
#             correct = (track_pred[mask] == c).sum() if total > 0 else 0
#             per_class_accuracy[c] = correct / total if total > 0 else None

#         per_class_df = pd.DataFrame.from_dict(
#             per_class_accuracy,
#             orient='index',
#             columns=['accuracy']
#         )

#         return {
#             'overall accuracy': overall_acc,
#             'per class accuracy': per_class_df
#         }
    
# # Main
# feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# X_seq, y_seq, df_tracks = build_track_sequences(
#     df=df,
#     feature_cols=feature_cols,
#     label_col='label_seg',
#     group_col='filename'
# )


# track_names = df_tracks[GROUP_COL].to_numpy()

# X_train_seq, X_test_seq, y_train_track_raw, y_test_track_raw, groups_train, groups_test = split_sequences(
#         groups=track_names,
#         X_seq=X_seq,
#         y_track=y_seq,
#         test_size=0.2,
#         random_state=12345
#     )

# # --------------------------------------------------------
# # 6.3 Encode track-level labels for deep model
# # --------------------------------------------------------
# le = LabelEncoder()
# y_train_enc = le.fit_transform(y_train_track_raw)
# y_test_enc = le.transform(y_test_track_raw)

# num_classes = len(le.classes_)

# y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
# y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)

# # Build df_train_tracks / df_test_tracks for deep model evaluation
# df_train_tracks = df_tracks[df_tracks[GROUP_COL].isin(groups_train)].reset_index(drop=True)
# df_test_tracks = df_tracks[df_tracks[GROUP_COL].isin(groups_test)].reset_index(drop=True)

# # --------------------------------------------------------
# # 6.4 Baseline Logistic Regression (segment-level, aggregated)
# #     using the same track split
# # --------------------------------------------------------
# # print("\n=== Baseline Logistic Regression (segment-level, aggregated) ===")
# # logistic_baseline = LogisticSegmentModelWithFixedSplit(
# #     df=df,
# #     train_tracks=groups_train,
# #     test_tracks=groups_test,
# #     label_col=TARGET_COL,
# #     group_col=GROUP_COL
# # )
# # baseline_results = logistic_baseline.fit()
# # print("Overall accuracy:", baseline_results["overall accuracy"])
# # print("Per-class accuracy:\n", baseline_results["per class accuracy"])

# # # --------------------------------------------------------
# # # 6.5 Logistic Regression with Feature Engineering (segment-level, aggregated)
# # #     using the same track split
# # # --------------------------------------------------------
# # print("\n=== Logistic Regression with Feature Engineering (segment-level, aggregated) ===")
# # logistic_fe = LogisticSegmentModelWithFE(
# #     df=df,
# #     train_tracks=groups_train,
# #     test_tracks=groups_test,
# #     label_col=TARGET_COL,
# #     group_col=GROUP_COL,
# #     poly_degree=2
# # )
# # fe_results = logistic_fe.fit()
# # print("Overall accuracy:", fe_results["overall accuracy"])
# # print("Per-class accuracy:\n", fe_results["per class accuracy"])

# # --------------------------------------------------------
# # 6.6 Deep Learning Model (track-level using segment sequences)
# # --------------------------------------------------------
# print("\n=== Deep Learning Model (track-level, sequence-based) ===")
# deep_model = TrackLevelDeepModel(num_classes=num_classes)

# history = deep_model.fit(
#     X_train_seq=X_train_seq,
#     y_train=y_train_cat,
#     X_test_seq=X_test_seq,
#     y_test=y_test_cat,
#     epochs=20
# )

# # Optionally save model
# deep_model.save("track_classifier")

# # Evaluate on test set (track-level)
# deep_results = deep_model.evaluate_tracks(
#     df_subset=df_test_tracks,
#     X_seq=X_test_seq,
#     y_true_track=y_test_enc,
#     groups_col=GROUP_COL
# )
# print("Overall accuracy:", deep_results['overall accuracy'])
# print("Per-class accuracy:\n", deep_results['per class accuracy'])

# import os
# import re
# import numpy as np
# import pandas as pd
# from glob import glob
# from PIL import Image

# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # ============================================================
# # 1. Configuration
# # ============================================================

# IMAGES_ROOT = r"C:\Users\apass\Downloads\gtzan\Data\images_original"

# # We will reuse these names so they plug into your existing DL code:
# TARGET_COL = 'label_seg'   # will hold genre label (for consistency)
# GROUP_COL = 'filename'     # track identifier, e.g. "blues.00000.wav"

# # ============================================================
# # 2. Helper to derive track "filename" from image path
# # ============================================================

# def image_path_to_filename_and_label(img_path):
#     """
#     For a path like:
#         images_original/blues/blues00000.png
#     We extract:
#         genre   = 'blues'   (the parent folder)
#         base    = 'blues00000'
#         filename= 'blues00000.wav'  (no dot, matches the base)
#     """
#     genre = os.path.basename(os.path.dirname(img_path))  # parent folder
#     base = os.path.splitext(os.path.basename(img_path))[0]
#     filename = base + ".wav"
#     return filename, genre


# # ============================================================
# # 3. Load mel spectrogram images into a "segment-level" dataframe
# # ============================================================

# def load_mel_images_as_segments(images_root=IMAGES_ROOT,
#                                 group_col=GROUP_COL,
#                                 label_col=TARGET_COL):
#     print("=== Debug: load_mel_images_as_segments ===")
#     print("Current working directory:", os.getcwd())
#     print("Absolute images_root     :", os.path.abspath(images_root))

#     # Recursive search for png/jpg/jpeg under images_root
#     pattern_png = os.path.join(images_root, '**', '*.png')
#     pattern_jpg = os.path.join(images_root, '**', '*.jpg')
#     pattern_jpeg = os.path.join(images_root, '**', '*.jpeg')

#     png_paths = glob(pattern_png, recursive=True)
#     jpg_paths = glob(pattern_jpg, recursive=True)
#     jpeg_paths = glob(pattern_jpeg, recursive=True)

#     print(f"PNG glob pattern:  {pattern_png}, found: {len(png_paths)}")
#     print(f"JPG glob pattern:  {pattern_jpg}, found: {len(jpg_paths)}")
#     print(f"JPEG glob pattern: {pattern_jpeg}, found: {len(jpeg_paths)}")

#     # Decide which set of files to use
#     if png_paths:
#         all_paths = png_paths
#         print("Using PNG files.")
#     elif jpg_paths:
#         all_paths = jpg_paths
#         print("Using JPG files.")
#     elif jpeg_paths:
#         all_paths = jpeg_paths
#         print("Using JPEG files.")
#     else:
#         print("WARNING: No .png/.jpg/.jpeg files found under:", images_root)
#         print("Expected something like: images_original/<genre>/blues00000.png")
#         return pd.DataFrame(), []

#     print("Example paths:", all_paths[:5])

#     records = []
#     X_img = []

#     for p in all_paths:
#         filename, genre = image_path_to_filename_and_label(p)

#         img = Image.open(p).convert('L')  # grayscale
#         arr = np.array(img, dtype=np.float32) / 255.0

#         records.append({group_col: filename, label_col: genre})
#         X_img.append(arr)

#     df_mel = pd.DataFrame(records)
#     print("Loaded mel segments:", len(df_mel))
#     print("df_mel columns:", df_mel.columns.tolist())
#     print(df_mel.head())

#     return df_mel, X_img


# df_mel, X_img = load_mel_images_as_segments(IMAGES_ROOT)

# if df_mel.empty:
#     raise RuntimeError(
#         "No mel images loaded (df_mel is empty). "
#         "Check IMAGES_ROOT, folder structure, and file extensions."
#     )


# # ============================================================
# # 4. Build per-track sequences from mel images
# # ============================================================

# def build_mel_track_sequences(df_mel, X_img,
#                               label_col=TARGET_COL,
#                               group_col=GROUP_COL):
#     """
#     Build per-track sequences from mel spectrogram images.

#     This function is analogous to build_track_sequences for tabular data,
#     but here each "segment" is a mel spectrogram image. We will:
#     - Group images by filename (track),
#     - Sort them lexicographically by their path order,
#     - Flatten each image into a 1D feature vector,
#     - Build sequences of shape (T_i, F_mel).

#     Parameters
#     ----------
#     df_mel : pd.DataFrame
#         Must contain [group_col, label_col] per image.
#     X_img : list of np.ndarray
#         List of mel spectrogram arrays (each item aligned to df_mel row).

#     Returns
#     -------
#     X_seq_mel : list of np.ndarray
#         Each element is a sequence of shape (T_i, F_mel) for track i.
#     y_seq_mel : np.ndarray
#         Track-level labels, one per track.
#     df_tracks_mel : pd.DataFrame
#         One row per track (metadata).
#     """
#     # Attach image index to df_mel to align properly
#     df_mel = df_mel.copy()
#     df_mel['img_idx'] = np.arange(len(df_mel))

#     # Group by track (filename)
#     track_names = df_mel[group_col].unique()

#     X_seq_mel = []
#     y_seq_mel = []
#     df_tracks_rows = []

#     for fname in track_names:
#         dfg = df_mel[df_mel[group_col] == fname].copy()

#         # Sort images by img_idx (original discovery order)
#         dfg = dfg.sort_values('img_idx')

#         # Collect images for this track
#         seg_arrays = []
#         for idx in dfg['img_idx']:
#             arr = X_img[idx]  # shape (H, W)
#             # Flatten to 1D feature vector, you can keep 2D if you later change the model
#             seg_arrays.append(arr.flatten())

#         # Shape: (T_i, F_mel)
#         seq = np.stack(seg_arrays, axis=0)
#         X_seq_mel.append(seq)

#         # Track-level label: assume single genre per track
#         y_seq_mel.append(dfg[label_col].iloc[0])

#         # Representative row for this track
#         df_tracks_rows.append(dfg.iloc[0])

#     df_tracks_mel = pd.DataFrame(df_tracks_rows)

#     return X_seq_mel, np.array(y_seq_mel), df_tracks_mel


# # ============================================================
# # 5. Track-level split for mel sequences (same logic as before)
# # ============================================================

# def split_mel_sequences(df_tracks_mel, X_seq_mel, y_seq_mel,
#                         test_size=0.2, random_state=12345,
#                         group_col=GROUP_COL):
#     """
#     Apply a track-level split to mel-based sequences.

#     Returns
#     -------
#     X_train_seq_mel, X_test_seq_mel,
#     y_train_mel_raw, y_test_mel_raw,
#     groups_train_mel, groups_test_mel
#     """
#     track_names = df_tracks_mel[group_col].to_numpy()

#     gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
#     idx_train, idx_test = next(gss.split(X_seq_mel, y=y_seq_mel, groups=track_names))

#     X_train_seq_mel = [X_seq_mel[i] for i in idx_train]
#     X_test_seq_mel = [X_seq_mel[i] for i in idx_test]

#     y_train_mel_raw = y_seq_mel[idx_train]
#     y_test_mel_raw = y_seq_mel[idx_test]

#     groups_train_mel = track_names[idx_train]
#     groups_test_mel = track_names[idx_test]

#     return (
#         X_train_seq_mel,
#         X_test_seq_mel,
#         y_train_mel_raw,
#         y_test_mel_raw,
#         groups_train_mel,
#         groups_test_mel
#     )


# # ============================================================
# # 6. Label encoding and padding for the existing DL model
# # ============================================================

# def prepare_mel_for_deep_model(X_train_seq_mel, X_test_seq_mel,
#                                y_train_mel_raw, y_test_mel_raw):
#     """
#     Encode labels and pad sequences to be ready for TrackLevelDeepModel.

#     This mirrors what you do for the tabular data, reusing:
#     - y_train_cat, y_test_cat
#     - num_classes

#     Returns
#     -------
#     X_train_seq_mel, X_test_seq_mel : list of np.ndarray
#         (unchanged, you will pass these into deep_model.fit)
#     y_train_cat_mel, y_test_cat_mel : np.ndarray (N, num_classes)
#     num_classes_mel : int
#     label_encoder_mel : LabelEncoder
#     """
#     le_mel = LabelEncoder()
#     y_train_enc_mel = le_mel.fit_transform(y_train_mel_raw)
#     y_test_enc_mel = le_mel.transform(y_test_mel_raw)

#     num_classes_mel = len(le_mel.classes_)
#     y_train_cat_mel = to_categorical(y_train_enc_mel, num_classes=num_classes_mel)
#     y_test_cat_mel = to_categorical(y_test_enc_mel, num_classes=num_classes_mel)

#     return (
#         X_train_seq_mel,
#         X_test_seq_mel,
#         y_train_cat_mel,
#         y_test_cat_mel,
#         num_classes_mel,
#         le_mel
#     )

# # ============================================================
# # 7. Example usage (can be integrated into your main script)
# # ============================================================
# if __name__ == "__main__":
#     # --------------------------------------------------------
#     # 7.1 Load mel images into a segment-like dataframe
#     # --------------------------------------------------------
#     df_mel, X_img = load_mel_images_as_segments(IMAGES_ROOT)

#     print("Mel segments loaded:", len(df_mel))
#     print("Example row:\n", df_mel.head())

#     # --------------------------------------------------------
#     # 7.2 Build per-track sequences for mel data
#     # --------------------------------------------------------
#     X_seq_mel, y_seq_mel, df_tracks_mel = build_mel_track_sequences(
#         df_mel=df_mel,
#         X_img=X_img,
#         label_col=TARGET_COL,
#         group_col=GROUP_COL
#     )

#     print("Number of mel tracks:", len(X_seq_mel))
#     print("Example sequence shape (T_i, F_mel):", X_seq_mel[0].shape)

#     # --------------------------------------------------------
#     # 7.3 Split mel data at track level (same style as tabular split)
#     # --------------------------------------------------------
#     (
#         X_train_seq_mel,
#         X_test_seq_mel,
#         y_train_mel_raw,
#         y_test_mel_raw,
#         groups_train_mel,
#         groups_test_mel
#     ) = split_mel_sequences(
#         df_tracks_mel=df_tracks_mel,
#         X_seq_mel=X_seq_mel,
#         y_seq_mel=y_seq_mel,
#         test_size=0.2,
#         random_state=12345,
#         group_col=GROUP_COL
#     )

#     # --------------------------------------------------------
#     # 7.4 Prepare labels for TrackLevelDeepModel
#     # --------------------------------------------------------
#     (
#         X_train_seq_mel,
#         X_test_seq_mel,
#         y_train_cat_mel,
#         y_test_cat_mel,
#         num_classes_mel,
#         le_mel
#     ) = prepare_mel_for_deep_model(
#         X_train_seq_mel=X_train_seq_mel,
#         X_test_seq_mel=X_test_seq_mel,
#         y_train_mel_raw=y_train_mel_raw,
#         y_test_mel_raw=y_test_mel_raw
#     )

#     print("Mel num_classes:", num_classes_mel)

#     # --------------------------------------------------------
#     # 7.5 Now you can instantiate your existing TrackLevelDeepModel
#     #     (from music_genre_predictor.py) and train it on mel data
#     # --------------------------------------------------------

#     deep_model_mel = TrackLevelDeepModel(num_classes=num_classes_mel)
#     history_mel = deep_model_mel.fit(
#         X_train_seq=X_train_seq_mel,
#         y_train=y_train_cat_mel,
#         X_test_seq=X_test_seq_mel,
#         y_test=y_test_cat_mel,
#         epochs=20
#     )