"""Sentence-to-document and document-to-daily stance aggregation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

SCORE_COLUMNS = ["dovish_score", "hawkish_score", "neutral_score"]


def aggregate_documents(
    sentences: pd.DataFrame,
    thresholds: Iterable[float] = (0.0, 0.5, 0.8),
) -> pd.DataFrame:
    """Aggregate classifier probabilities to one row per ECB document."""
    required = {"doc_id", "pub_date", "doc_type", *SCORE_COLUMNS}
    missing = sorted(required - set(sentences.columns))
    if missing:
        raise ValueError(f"Missing sentence columns: {missing}")
    frame = sentences.copy()
    scores = frame[SCORE_COLUMNS].to_numpy(dtype=float)
    labels = np.array(["dovish", "hawkish", "neutral"])
    frame["max_class"] = labels[scores.argmax(axis=1)]
    frame["max_prob"] = scores.max(axis=1)

    rows: list[dict[str, object]] = []
    for doc_id, group in frame.groupby("doc_id", sort=False):
        n_total = len(group)
        row: dict[str, object] = {
            "doc_id": doc_id,
            "pub_date": group["pub_date"].iloc[0],
            "doc_type": group["doc_type"].iloc[0],
            "speaker": group["speaker"].iloc[0] if "speaker" in group else "",
            "n_total": n_total,
        }
        for threshold in thresholds:
            label = f"t{int(round(threshold * 100)):03d}"
            eligible = group["max_prob"] >= threshold
            dovish = eligible & group["max_class"].eq("dovish")
            hawkish = eligible & group["max_class"].eq("hawkish")
            row[f"P_doc_dovish_{label}"] = group.loc[dovish, "dovish_score"].sum() / n_total
            row[f"P_doc_hawkish_{label}"] = group.loc[hawkish, "hawkish_score"].sum() / n_total
            row[f"policy_intensity_{label}"] = (dovish.sum() + hawkish.sum()) / n_total
        rows.append(row)
    out = pd.DataFrame(rows)
    out["pub_date"] = pd.to_datetime(out["pub_date"], errors="coerce")
    return out.sort_values(["pub_date", "doc_id"]).reset_index(drop=True)


def build_daily_event_series(
    documents: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    source_filter: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Equal-weight documents on event days and reindex to calendar frequency."""
    frame = documents.copy()
    if source_filter is not None:
        frame = frame[frame["doc_type"].isin(source_filter)]
    frame["pub_date"] = pd.to_datetime(frame["pub_date"], errors="coerce")
    score_cols = [
        col
        for col in frame.columns
        if col.startswith(("P_doc_dovish_", "P_doc_hawkish_", "policy_intensity_"))
    ]
    by_day = frame.groupby(frame["pub_date"].dt.normalize())[score_cols].mean()
    counts = frame.groupby(frame["pub_date"].dt.normalize()).size()
    calendar = pd.date_range(start, end, freq="D")
    daily = by_day.reindex(calendar)
    daily.index.name = "date"
    daily["is_event_day"] = daily[score_cols[0]].notna().astype(int)
    daily["n_docs_today"] = counts.reindex(calendar).fillna(0).astype(int)
    return daily.reset_index()


def persist_to_next_event(daily: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Forward-fill event values without backfilling before the first event."""
    out = daily.copy()
    for column in columns:
        out[f"{column}_persist"] = out[column].ffill()
    return out


def rescale_stance(
    daily: pd.DataFrame,
    dovish_column: str,
    hawkish_column: str,
    insample_start: str | pd.Timestamp,
    insample_end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Apply the Sadik in-sample-maximum scaling convention."""
    out = daily.copy()
    dates = pd.to_datetime(out["date"])
    mask = dates.between(pd.Timestamp(insample_start), pd.Timestamp(insample_end))
    dovish_max = out.loc[mask, dovish_column].max()
    hawkish_max = out.loc[mask, hawkish_column].max()
    if not np.isfinite(dovish_max) or dovish_max <= 0:
        raise ValueError("Dovish in-sample maximum must be positive.")
    if not np.isfinite(hawkish_max) or hawkish_max <= 0:
        raise ValueError("Hawkish in-sample maximum must be positive.")
    out["P_t"] = out[dovish_column] / dovish_max
    out["N_t"] = -out[hawkish_column] / hawkish_max
    out["S_t"] = out["P_t"] + out["N_t"]
    return out
