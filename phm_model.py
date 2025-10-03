"""
PHM Wafer Anomaly Detection model
=================================

This module contains a complete training and inference pipeline for
addressing the fault‑detection problem posed in the HW3 dataset.  The
goal is to build a binary classifier that can decide whether a wafer
fabrication process run will ``Pass`` (label ``0``) or ``Fail``
(``1``).  The chosen approach uses a regularised logistic regression
classifier together with straightforward feature engineering and
median imputation.  The model was selected after extensive
experimentation (summarised in the accompanying notebook) which
indicated that a linear model with class weighting and a carefully
chosen probability threshold offered the best macro F1 score on the
provided training data.

Key steps in the pipeline are:

1.  Loading the training and test CSV files supplied for HW3.
2.  Converting the ``Time`` column into several numeric features
    (Unix timestamp and calendar components) and dropping the original
    text column along with the ``Id`` column.
3.  Filling missing values with the median of each feature computed
    from the training data.  This avoids discarding any sensors while
    ensuring the model sees numerically sensible values.
4.  Standardising the features (zero mean and unit variance) using a
    ``StandardScaler`` fitted on the training data.  The same scaler
    is applied to the test data.
5.  Training a logistic regression classifier with an L2 penalty and
    ``class_weight='balanced'`` to counter the severe class
    imbalance.  The solver automatically handles high‑dimensional
    sparse problems.
6.  Generating probabilistic predictions for the test set and
    converting them into binary labels using a custom threshold.  A
    threshold of ``0.29`` was determined via cross‑validation to
    maximise the macro F1 score on the training data (see the
    notebook for details).
7.  Writing out a submission CSV that matches the sample submission
    format, containing two columns: ``Id`` and ``Pass/Fail``.

Usage example::

    python phm_model.py --train hw3_train.csv --test hw3_test.csv \
                        --output submission.csv

Requirements: pandas, numpy and scikit‑learn must be available in the
Python environment.  These packages are installed in the execution
environment used for this project.
"""

import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def preprocess_dataframe(df: pd.DataFrame, drop_target: bool = False) -> pd.DataFrame:
    """Expand the ``Time`` column, drop identifiers and optionally the target.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataframe.
    drop_target : bool, optional
        When ``True``, the ``Pass/Fail`` column will be removed.  Use
        this for the training set where the target is stored
        separately.

    Returns
    -------
    pandas.DataFrame
        A numeric feature matrix with engineered time features and no
        identifier columns.  All remaining columns are coerced to
        numeric types; invalid parsing results in ``NaN`` values.
    """
    df = df.copy()
    # Convert Time column to datetime; invalid entries become NaT
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    # Derive numeric time features
    df["Time_timestamp"] = df["Time"].values.astype("int64") // 10 ** 9
    df["Time_month"] = df["Time"].dt.month
    df["Time_day"] = df["Time"].dt.day
    df["Time_hour"] = df["Time"].dt.hour
    df["Time_minute"] = df["Time"].dt.minute
    df["Time_weekday"] = df["Time"].dt.weekday
    df.drop(columns=["Time"], inplace=True)
    # Remove identifiers
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)
    # Optionally remove the target label
    if drop_target and "Pass/Fail" in df.columns:
        df.drop(columns=["Pass/Fail"], inplace=True)
    # Convert all remaining columns to numeric values
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def fit_scaler_and_impute(X: pd.DataFrame) -> Tuple[StandardScaler, pd.Series]:
    """Compute median values and fit a StandardScaler on the training data.

    Parameters
    ----------
    X : pandas.DataFrame
        Training feature matrix.

    Returns
    -------
    tuple
        A tuple containing the fitted StandardScaler and a Series of
        median values for each column.  The scaler and medians should
        later be applied to transform new data.
    """
    # Compute column medians for imputation
    medians = X.median()
    # Impute missing values with the median
    X_imputed = X.fillna(medians)
    # Fit a scaler to the imputed training data
    scaler = StandardScaler()
    scaler.fit(X_imputed)
    return scaler, medians


def impute_and_scale(X: pd.DataFrame, medians: pd.Series, scaler: StandardScaler) -> np.ndarray:
    """Impute missing values and standardise features using precomputed statistics.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix to transform.
    medians : pandas.Series
        Per‑column median values computed on the training data.
    scaler : StandardScaler
        Fitted scaler for standardising features.

    Returns
    -------
    numpy.ndarray
        The imputed and scaled feature matrix ready for model consumption.
    """
    X_imputed = X.fillna(medians)
    return scaler.transform(X_imputed)


def train_model(X_scaled: np.ndarray, y: pd.Series) -> LogisticRegression:
    """Train a logistic regression classifier on the scaled features.

    The classifier uses L2 regularisation and balanced class weights to
    address the class imbalance present in the dataset.

    Parameters
    ----------
    X_scaled : numpy.ndarray
        Scaled training feature matrix.
    y : pandas.Series
        Target labels (0 or 1).

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A fitted classifier.
    """
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="lbfgs"
    )
    clf.fit(X_scaled, y)
    return clf


def predict_with_threshold(model: LogisticRegression, X_scaled: np.ndarray, threshold: float) -> np.ndarray:
    """Generate binary predictions using a custom probability threshold.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        Trained classifier.
    X_scaled : numpy.ndarray
        Scaled feature matrix for which to make predictions.
    threshold : float
        Probability cutoff; observations with predicted probability >=
        threshold are labelled as the positive class (1).

    Returns
    -------
    numpy.ndarray
        Array of binary predictions (0 or 1).
    """
    prob = model.predict_proba(X_scaled)[:, 1]
    return (prob >= threshold).astype(int)


def main(args: argparse.Namespace) -> None:
    """Main entry point for the command line interface.

    This function orchestrates the data loading, preprocessing,
    training and prediction steps.  It reports the number of samples
    and predicted failing runs, and writes the submission file.
    """
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Drop rows lacking a label (there is one blank row in the train set)
    train_df = train_df.dropna(subset=["Pass/Fail"])

    # Extract target
    y = train_df["Pass/Fail"].astype(int)

    # Preprocess features
    X_train = preprocess_dataframe(train_df, drop_target=True)
    X_test = preprocess_dataframe(test_df, drop_target=False)

    # Fit imputer and scaler on training data
    scaler, medians = fit_scaler_and_impute(X_train)

    # Transform both training and test sets
    X_train_scaled = impute_and_scale(X_train, medians, scaler)
    X_test_scaled = impute_and_scale(X_test, medians, scaler)

    # Train classifier
    model = train_model(X_train_scaled, y)
    # Use threshold found via cross‑validation (see notebook)

    threshold = 0.29
    predictions = predict_with_threshold(model, X_test_scaled, threshold)
    
    # Prepare submission dataframe
    submission = pd.DataFrame({
        "Id": test_df["Id"].astype(int),
        "Pass/Fail": predictions
    })
    submission.to_csv(args.output, index=False)

    # Report summary statistics
    num_samples = X_train.shape[0]
    num_features = X_train.shape[1]
    num_fail = int(predictions.sum())
    print(f"Model trained on {num_samples} samples with {num_features} features.")
    print(f"Using threshold {threshold}, predicted {num_fail} failing runs out of {len(predictions)}.")
    print(f"Submission file written to {args.output}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PHM fault detection model using logistic regression and generate predictions."
    )
    parser.add_argument("--train", required=True, help="Path to the training CSV file (hw3_train.csv)")
    parser.add_argument("--test", required=True, help="Path to the test CSV file (hw3_test.csv)")
    parser.add_argument("--output", required=True, help="Path to write the submission CSV file")
    main(parser.parse_args())