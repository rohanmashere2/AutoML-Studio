"""
Automated Feature Engineering Module
=====================================

Provides intelligent, automated feature generation from a cleaned DataFrame.
Each engineering step is independently guarded so that a failure in one step
does not prevent the remaining steps from executing.

Steps (executed in order):
    1. DateTime Feature Extraction
    2. Log Transform for Skewed Features
    3. Interaction Features (pairwise products)
    4. Binning for Continuous Features
    5. Text Length Features
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_engineer_features(df, profile, target_col=None):
    """
    Automatically generate new features from the dataset.

    Args:
        df: pandas DataFrame (cleaned, before transform).
        profile: dict from profiler with column info.
        target_col: name of target column (excluded from engineering).

    Returns:
        tuple: (engineered_df, engineering_report)
            - engineered_df: DataFrame with new features appended.
            - engineering_report: dict describing every step and a summary.
    """
    logger.info("Starting automated feature engineering (%d cols, %d rows)",
                df.shape[1], df.shape[0])

    engineered_df = df.copy()
    original_feature_count = engineered_df.shape[1]
    steps = []

    # 1. DateTime Feature Extraction
    step = _extract_datetime_features(engineered_df, profile, target_col)
    steps.append(step)

    # 2. Log Transform for Skewed Features
    step = _log_transform_skewed(engineered_df, profile, target_col)
    steps.append(step)

    # 3. Interaction Features
    step = _create_interaction_features(engineered_df, profile, target_col)
    steps.append(step)

    # 4. Binning for Continuous Features
    step = _bin_continuous_features(engineered_df, profile, target_col)
    steps.append(step)

    # 5. Text Length Features
    step = _text_length_features(engineered_df, profile, target_col)
    steps.append(step)

    total_created = sum(s.get("count", 0) for s in steps)
    report = {
        "steps": steps,
        "summary": {
            "total_features_created": total_created,
            "original_features": original_feature_count,
            "final_features": engineered_df.shape[1],
        },
    }

    logger.info(
        "Feature engineering complete — %d new features (original %d → final %d)",
        total_created,
        original_feature_count,
        engineered_df.shape[1],
    )
    return engineered_df, report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_target(col, target_col):
    """Return True if *col* is the target column and should be skipped."""
    return target_col is not None and col == target_col


def _get_candidate_columns(df, profile, target_col):
    """Return lists of numeric and object columns, excluding the target."""
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if not _is_target(c, target_col)
    ]
    object_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if not _is_target(c, target_col)
    ]
    return numeric_cols, object_cols


def _make_step(name, icon, description, count, features_created, applied):
    """Build a standardised step-report dict."""
    return {
        "name": name,
        "icon": icon,
        "description": description,
        "count": count,
        "features_created": list(features_created),
        "applied": applied,
    }


# ---------------------------------------------------------------------------
# Step 1 – DateTime Feature Extraction
# ---------------------------------------------------------------------------

def _extract_datetime_features(df, profile, target_col):
    """Extract calendar / clock components from datetime columns."""
    step_name = "DateTime Features"
    icon = "📅"
    features_created = []

    try:
        datetime_cols = _detect_datetime_columns(df, profile, target_col)

        for col in datetime_cols:
            try:
                dt_series = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                logger.warning("Could not parse column '%s' as datetime", col)
                continue

            # Year, month, day, day_of_week, quarter
            for attr, suffix in [
                ("year", "year"),
                ("month", "month"),
                ("day", "day"),
                ("dayofweek", "day_of_week"),
                ("quarter", "quarter"),
            ]:
                feat_name = f"{col}_{suffix}"
                df[feat_name] = getattr(dt_series.dt, attr)
                features_created.append(feat_name)

            # Hour – only when time information is present
            if _has_time_component(dt_series):
                feat_name = f"{col}_hour"
                df[feat_name] = dt_series.dt.hour
                features_created.append(feat_name)

            # is_weekend (Saturday=5, Sunday=6)
            feat_name = f"{col}_is_weekend"
            df[feat_name] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
            features_created.append(feat_name)

            # Drop original datetime column
            df.drop(columns=[col], inplace=True)
            logger.info("Extracted %d features from datetime column '%s'",
                        len([f for f in features_created if f.startswith(col)]),
                        col)

        count = len(features_created)
        description = (
            f"Extracted {count} features from {len(datetime_cols)} datetime column(s)"
            if datetime_cols
            else "No datetime columns detected"
        )
        return _make_step(step_name, icon, description, count,
                          features_created, bool(datetime_cols))

    except Exception as exc:
        logger.error("DateTime feature extraction failed: %s", exc, exc_info=True)
        return _make_step(step_name, icon, f"Failed: {exc}", 0, [], False)


def _detect_datetime_columns(df, profile, target_col):
    """Identify columns that are (or should be) datetime."""
    dt_cols = []

    # 1. Already datetime dtype
    for col in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        if not _is_target(col, target_col) and col not in dt_cols:
            dt_cols.append(col)

    # 2. Flagged by the profiler
    if isinstance(profile, dict):
        columns_info = profile.get("columns", profile.get("column_stats", {}))
        if isinstance(columns_info, dict):
            for col, info in columns_info.items():
                if _is_target(col, target_col) or col not in df.columns:
                    continue
                col_type = ""
                if isinstance(info, dict):
                    col_type = str(info.get("type", info.get("dtype", ""))).lower()
                if "date" in col_type or "time" in col_type:
                    if col not in dt_cols:
                        dt_cols.append(col)

    # 3. Object columns that look like dates (heuristic parse)
    for col in df.select_dtypes(include=["object"]).columns:
        if _is_target(col, target_col) or col in dt_cols:
            continue
        try:
            sample = df[col].dropna().head(20)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                dt_cols.append(col)
        except Exception:
            pass

    return dt_cols


def _has_time_component(dt_series):
    """Return True if any non-null entry has a non-midnight time."""
    try:
        non_null = dt_series.dropna()
        if non_null.empty:
            return False
        sample = non_null.head(100)
        return (sample.dt.hour != 0).any() or (sample.dt.minute != 0).any()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 2 – Log Transform for Skewed Features
# ---------------------------------------------------------------------------

def _log_transform_skewed(df, profile, target_col):
    """Add log1p features for highly skewed numeric columns."""
    step_name = "Log Transform"
    icon = "📐"
    features_created = []

    try:
        numeric_cols, _ = _get_candidate_columns(df, profile, target_col)

        for col in numeric_cols:
            try:
                series = df[col].dropna()
                if series.empty:
                    continue

                # Skip columns that contain negative values
                if (series < 0).any():
                    logger.debug("Skipping log transform for '%s' (negative values)", col)
                    continue

                skewness = series.skew()
                if skewness > 2:
                    feat_name = f"{col}_log"
                    df[feat_name] = np.log1p(df[col])
                    features_created.append(feat_name)
                    logger.debug("Created log feature for '%s' (skew=%.2f)", col, skewness)
            except Exception as inner_exc:
                logger.warning("Log transform failed for '%s': %s", col, inner_exc)

        count = len(features_created)
        description = (
            f"Created {count} log-transformed feature(s)"
            if count
            else "No highly skewed (>2) numeric columns found"
        )
        return _make_step(step_name, icon, description, count,
                          features_created, count > 0)

    except Exception as exc:
        logger.error("Log transform step failed: %s", exc, exc_info=True)
        return _make_step(step_name, icon, f"Failed: {exc}", 0, [], False)


# ---------------------------------------------------------------------------
# Step 3 – Interaction Features
# ---------------------------------------------------------------------------

def _create_interaction_features(df, profile, target_col):
    """Create pairwise product features for the top-variance numeric columns."""
    step_name = "Interaction Features"
    icon = "🔗"
    features_created = []

    try:
        numeric_cols, _ = _get_candidate_columns(df, profile, target_col)

        # Coefficient of variation (std / mean) – higher means more informative
        cv_scores = {}
        for col in numeric_cols:
            try:
                mean = df[col].mean()
                std = df[col].std()
                if mean != 0 and pd.notna(mean) and pd.notna(std):
                    cv_scores[col] = abs(std / mean)
            except Exception:
                pass

        # Pick top-5 by CV
        top_cols = sorted(cv_scores, key=cv_scores.get, reverse=True)[:5]

        if len(top_cols) < 2:
            return _make_step(step_name, icon,
                              "Fewer than 2 eligible numeric columns", 0, [], False)

        # Generate pairs, keep only top 3
        pairs = list(combinations(top_cols, 2))[:3]

        for col_a, col_b in pairs:
            try:
                feat_name = f"{col_a}_x_{col_b}"
                df[feat_name] = df[col_a] * df[col_b]
                features_created.append(feat_name)
            except Exception as inner_exc:
                logger.warning("Interaction '%s' x '%s' failed: %s",
                               col_a, col_b, inner_exc)

        count = len(features_created)
        description = f"Created {count} interaction feature(s) from top-CV columns"
        return _make_step(step_name, icon, description, count,
                          features_created, count > 0)

    except Exception as exc:
        logger.error("Interaction features step failed: %s", exc, exc_info=True)
        return _make_step(step_name, icon, f"Failed: {exc}", 0, [], False)


# ---------------------------------------------------------------------------
# Step 4 – Binning for Continuous Features
# ---------------------------------------------------------------------------

def _bin_continuous_features(df, profile, target_col):
    """Create quantile-binned versions of high-cardinality numeric columns."""
    step_name = "Quantile Binning"
    icon = "📊"
    features_created = []

    try:
        numeric_cols, _ = _get_candidate_columns(df, profile, target_col)

        for col in numeric_cols:
            try:
                n_unique = df[col].nunique()
                if n_unique <= 50:
                    continue

                feat_name = f"{col}_binned"
                df[feat_name] = pd.qcut(
                    df[col], q=5, labels=[0, 1, 2, 3, 4], duplicates="drop"
                )
                # Convert to int (nullable) to keep things tidy
                df[feat_name] = df[feat_name].astype("Int64")
                features_created.append(feat_name)
            except Exception as inner_exc:
                logger.warning("Binning failed for '%s': %s", col, inner_exc)

        count = len(features_created)
        description = (
            f"Binned {count} continuous column(s) into 5 quantiles"
            if count
            else "No numeric columns with >50 unique values"
        )
        return _make_step(step_name, icon, description, count,
                          features_created, count > 0)

    except Exception as exc:
        logger.error("Binning step failed: %s", exc, exc_info=True)
        return _make_step(step_name, icon, f"Failed: {exc}", 0, [], False)


# ---------------------------------------------------------------------------
# Step 5 – Text Length Features
# ---------------------------------------------------------------------------

def _text_length_features(df, profile, target_col):
    """Add word-count and char-count features for long-text columns."""
    step_name = "Text Length Features"
    icon = "📝"
    features_created = []

    try:
        _, object_cols = _get_candidate_columns(df, profile, target_col)

        for col in object_cols:
            try:
                str_series = df[col].astype(str)
                avg_len = str_series.str.len().mean()

                if pd.isna(avg_len) or avg_len <= 20:
                    continue

                word_feat = f"{col}_word_count"
                char_feat = f"{col}_char_count"

                df[word_feat] = str_series.str.split().str.len().fillna(0).astype(int)
                df[char_feat] = str_series.str.len().fillna(0).astype(int)

                features_created.extend([word_feat, char_feat])
                logger.debug("Text features for '%s' (avg len %.1f)", col, avg_len)
            except Exception as inner_exc:
                logger.warning("Text features failed for '%s': %s", col, inner_exc)

        count = len(features_created)
        description = (
            f"Created {count} text-length feature(s)"
            if count
            else "No text columns with avg length > 20"
        )
        return _make_step(step_name, icon, description, count,
                          features_created, count > 0)

    except Exception as exc:
        logger.error("Text length features step failed: %s", exc, exc_info=True)
        return _make_step(step_name, icon, f"Failed: {exc}", 0, [], False)
