from __future__ import annotations

import numpy as np
import pandas as pd

from ecb_vol_forecasting.nlp import (
    aggregate_documents,
    build_daily_event_series,
    persist_to_next_event,
    rescale_stance,
)


def test_document_aggregation_uses_total_sentence_denominator() -> None:
    sentences = pd.DataFrame(
        {
            "doc_id": [1, 1, 1],
            "pub_date": ["2020-01-01"] * 3,
            "doc_type": ["speech"] * 3,
            "speaker": ["A"] * 3,
            "dovish_score": [0.8, 0.1, 0.1],
            "hawkish_score": [0.1, 0.7, 0.1],
            "neutral_score": [0.1, 0.2, 0.8],
        }
    )
    result = aggregate_documents(sentences, thresholds=[0.5]).iloc[0]
    assert np.isclose(result["P_doc_dovish_t050"], 0.8 / 3)
    assert np.isclose(result["P_doc_hawkish_t050"], 0.7 / 3)
    assert np.isclose(result["policy_intensity_t050"], 2 / 3)


def test_persistence_starts_at_first_event() -> None:
    daily = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=4), "x": [np.nan, 0.3, np.nan, 0.7]}
    )
    result = persist_to_next_event(daily, ["x"])
    assert np.isnan(result.loc[0, "x_persist"])
    assert result["x_persist"].iloc[1:].tolist() == [0.3, 0.3, 0.7]


def test_daily_equal_weights_and_rescaling() -> None:
    docs = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "pub_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-04"]),
            "doc_type": ["speech"] * 3,
            "P_doc_dovish_t050": [0.2, 0.6, 0.8],
            "P_doc_hawkish_t050": [0.4, 0.2, 0.8],
            "policy_intensity_t050": [0.5, 0.5, 0.9],
        }
    )
    daily = build_daily_event_series(docs, "2020-01-01", "2020-01-05")
    assert np.isclose(
        daily.loc[daily["date"].eq(pd.Timestamp("2020-01-02")), "P_doc_dovish_t050"].iloc[0], 0.4
    )
    persisted = persist_to_next_event(daily, ["P_doc_dovish_t050", "P_doc_hawkish_t050"])
    result = rescale_stance(
        persisted,
        "P_doc_dovish_t050_persist",
        "P_doc_hawkish_t050_persist",
        "2020-01-01",
        "2020-01-05",
    )
    assert np.isclose(result["P_t"].max(), 1.0)
    assert np.isclose(result["N_t"].min(), -1.0)
