"""news_utils.py

Helpers to fetch news headlines (NewsAPI) and run sentiment analysis using
Hugging Face transformers. This module keeps the model instantiation lazy so the
app can still run if `transformers` is not installed; the endpoint will return a
helpful error in that case.

Environment variables:
- NEWSAPI_KEY: API key for NewsAPI.org (optional if using another source)

Functions:
- get_news_headlines_newsapi(api_key, query, limit)
- analyze_sentiment(texts)
"""
from __future__ import annotations

import os
import logging
import requests
from typing import List, Dict

logger = logging.getLogger(__name__)

# Lazy import holder for the transformers pipeline
_sentiment_pipeline = None


def get_news_headlines_newsapi(api_key: str, query: str, limit: int = 10, require_query_in_title: bool = True) -> List[str]:
    """Fetch headlines from NewsAPI.org. Returns list of headline strings.

    Raises RuntimeError on HTTP or parse errors.
    """
    url = "https://newsapi.org/v2/everything"
    # Prefer matches where the query appears in the title to avoid loose/irrelevant
    # results. NewsAPI supports `qInTitle` which narrows results to headlines
    # containing the query. If require_query_in_title is False we use the more
    # relaxed `q` parameter.
    if require_query_in_title:
        params = {"qInTitle": query, "language": "en", "pageSize": limit, "sortBy": "publishedAt"}
    else:
        params = {"q": query, "language": "en", "pageSize": limit, "sortBy": "publishedAt"}
    headers = {"Authorization": api_key}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.exception("Failed to fetch news from NewsAPI")
        raise RuntimeError(f"NewsAPI fetch failed: {e}")

    articles = data.get("articles", [])
    headlines = [a.get("title") or a.get("description") or "" for a in articles]

    # If we used the relaxed `q` parameter, filter client-side for titles that
    # actually contain the query to reduce unrelated hits (case-insensitive).
    if not require_query_in_title:
        lower_q = (query or "").lower()
        filtered = [h for h in headlines if lower_q in (h or "").lower()]
        if filtered:
            return filtered[:limit]

    return headlines[:limit]


def _init_sentiment_pipeline():
    """Initialize the Hugging Face sentiment pipeline (lazy)."""
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline

    try:
        from transformers import pipeline
    except Exception as e:  # pragma: no cover - external dependency
        raise RuntimeError("transformers package not installed: please pip install transformers and torch")

    # Use the default sentiment model
    _sentiment_pipeline = pipeline("sentiment-analysis")
    return _sentiment_pipeline


def analyze_sentiment(texts: List[str]) -> List[Dict[str, object]]:
    """Return a list of sentiment results for each text.

    Each result is a dict: { 'label': 'POSITIVE'|'NEGATIVE'|'NEUTRAL', 'score': float }
    If transformers is not installed this will raise RuntimeError with guidance.
    """
    if not texts:
        return []

    try:
        pipe = _init_sentiment_pipeline()
    except RuntimeError:
        # Re-raise with the same message
        raise

    # The pipeline accepts a list of strings
    results = pipe(texts)
    # Normalize results to dicts with label and score
    out = []
    for r in results:
        label = r.get("label")
        score = float(r.get("score", 0.0))
        out.append({"label": label, "score": score})
    return out
