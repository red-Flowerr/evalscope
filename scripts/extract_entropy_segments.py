#!/usr/bin/env python3
"""
Split model responses into high-entropy and low-entropy segments based on token entropies.
"""

import argparse
import html
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

WORDNET_AVAILABLE = False
try:
    from nltk.corpus import wordnet as wn  # type: ignore
    from nltk import download as nltk_download  # type: ignore
    WORDNET_AVAILABLE = True
except Exception:  # pragma: no cover - nltk not installed
    wn = None
    nltk_download = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Split responses into high- and low-entropy segments.')
    parser.add_argument('--input', required=True, help='Path to observations.jsonl')
    parser.add_argument('--output', required=True, help='Path to write segments.html')
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='Absolute entropy threshold. Overrides other threshold options if provided.'
    )
    parser.add_argument(
        '--percentile', type=float, default=None,
        help='Percentile (0-100) of token entropies to use as threshold (per response).'
    )
    parser.add_argument(
        '--skip-unknown', action='store_true',
        help='Skip tokens that have undefined entropy values instead of marking them as unknown segments.'
    )
    parser.add_argument(
        '--filter-synonym-high', action='store_true',
        help='Detect high-entropy tokens whose alternate predictions are synonyms and mark them separately.'
    )
    return parser.parse_args()


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return math.nan
    percentile = max(0.0, min(100.0, percentile))
    if percentile <= 0.0:
        return min(values)
    if percentile >= 100.0:
        return max(values)
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_vals[lower]
    fraction = rank - lower
    return sorted_vals[lower] * (1.0 - fraction) + sorted_vals[upper] * fraction


def derive_threshold(
    entropies: List[Optional[float]],
    metrics: Dict[str, Any],
    absolute: Optional[float],
    percentile: Optional[float],
) -> Tuple[float, str]:
    numeric = [value for value in entropies if isinstance(value, (int, float))]
    if absolute is not None:
        return float(absolute), 'manual'

    if percentile is not None and numeric:
        return _percentile(numeric, percentile), f'percentile_{percentile:g}'

    metric_value = metrics.get('mean_token_entropy')
    if isinstance(metric_value, (int, float)):
        return float(metric_value), 'mean_token_entropy'

    if numeric:
        return float(sum(numeric) / len(numeric)), 'auto_mean'

    return 0.0, 'default_zero'


def ensure_wordnet() -> None:
    global WORDNET_AVAILABLE
    if not WORDNET_AVAILABLE:
        return
    try:
        wn.synsets('example')
    except LookupError:  # pragma: no cover - first run
        if nltk_download is not None:
            nltk_download('wordnet')
            nltk_download('omw-1.4')
        try:
            wn.synsets('example')
        except LookupError:
            WORDNET_AVAILABLE = False


@lru_cache(maxsize=2048)
def normalize_token(token: str) -> str:
    return re.sub(r'[^a-zA-Z]+', '', token).lower()


@lru_cache(maxsize=2048)
def synonym_set(token: str) -> set:
    if not WORDNET_AVAILABLE:
        return set()
    ensure_wordnet()
    synonyms: set = set()
    if not WORDNET_AVAILABLE:
        return synonyms
    for syn in wn.synsets(token):
        for lemma in syn.lemmas():
            normalized = normalize_token(lemma.name())
            if normalized:
                synonyms.add(normalized)
    return synonyms


def extract_alternates(token_entry: Dict[str, Any]) -> List[str]:
    top = token_entry.get('top_logprobs')
    if isinstance(top, dict):
        return list(top.keys())
    if isinstance(top, list):
        alts = []
        for item in top:
            if isinstance(item, dict):
                val = item.get('token')
                if isinstance(val, str):
                    alts.append(val)
        return alts
    return []


def is_synonym_high(token_entry: Dict[str, Any]) -> bool:
    if not WORDNET_AVAILABLE:
        return False
    token_text = token_entry.get('text', '')
    norm_actual = normalize_token(token_text)
    if not norm_actual:
        return False
    alternates = extract_alternates(token_entry)
    if not alternates:
        return False
    synonyms_actual = synonym_set(norm_actual)
    for alt in alternates:
        norm_alt = normalize_token(str(alt))
        if not norm_alt or norm_alt == norm_actual:
            return True
        if norm_alt in synonyms_actual:
            return True
        if norm_actual in synonym_set(norm_alt):
            return True
    return False


def segment_tokens(
    tokens: Iterable[Dict[str, Any]],
    entropies: List[Optional[float]],
    categories: Optional[List[str]],
    threshold: float,
    skip_unknown: bool,
    filter_synonym_high: bool,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    current_tokens: List[str] = []
    current_entropies: List[Optional[float]] = []
    current_categories: List[str] = []
    current_type: Optional[str] = None
    start_index: Optional[int] = None
    last_index: int = -1

    def flush(end_index: int) -> None:
        nonlocal current_tokens, current_entropies, current_categories, current_type, start_index
        if not current_tokens or current_type is None or start_index is None:
            current_tokens = []
            current_entropies = []
            current_categories = []
            current_type = None
            start_index = None
            return

        segments.append({
            'type': current_type,
            'text': ''.join(current_tokens),
        })
        current_tokens = []
        current_entropies = []
        current_categories = []
        current_type = None
        start_index = None

    for idx, token in enumerate(tokens):
        last_index = idx
        entropy = entropies[idx] if idx < len(entropies) else None
        category = categories[idx] if categories and idx < len(categories) else 'unknown'

        if entropy is None:
            if skip_unknown:
                flush(idx - 1)
                continue
            seg_type = 'unknown'
        else:
            if entropy >= threshold:
                if filter_synonym_high and is_synonym_high(token):
                    seg_type = 'synonym'
                else:
                    seg_type = 'high'
            else:
                seg_type = 'low'

        token_text = token.get('text', '')
        if seg_type != current_type:
            flush(idx - 1)
            current_type = seg_type
            start_index = idx

        current_tokens.append(token_text)
        current_entropies.append(entropy)
        current_categories.append(category or 'unknown')

    if last_index >= 0:
        flush(last_index)
    return segments


def process_observations(
    input_path: Path,
    output_path: Path,
    threshold: Optional[float],
    percentile: Optional[float],
    skip_unknown: bool,
    filter_synonym_high: bool,
) -> None:
    samples: List[Dict[str, Any]] = []
    with input_path.open('r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metrics = record.get('metrics') or {}
            entropies = metrics.get('token_entropies') or []
            tokens = record.get('tokens') or []
            categories = record.get('token_categories')

            thresh_value, source = derive_threshold(entropies, metrics, threshold, percentile)
            segments = segment_tokens(
                tokens,
                entropies,
                categories,
                thresh_value,
                skip_unknown,
                filter_synonym_high,
            )

            samples.append({
                'id': record.get('id'),
                'threshold': thresh_value,
                'threshold_source': source,
                'segments': segments,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as writer:
        writer.write(
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Entropy Segments</title>"
            "<style>"
            "body{font-family:system-ui,-apple-system,\"Segoe UI\",sans-serif;background:#f5f6fb;color:#1f2530;margin:0;padding:24px;}"
            "h1{margin-top:0;font-size:26px;color:#182140;margin-bottom:24px;}"
            ".legend{display:flex;gap:12px;font-size:12px;color:#4a587a;background:#fff;border:1px solid #dfe4f6;"
            "border-radius:10px;padding:10px 14px;margin-bottom:24px;flex-wrap:wrap;}"
            ".legend span{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border-radius:6px;background:#eef1ff;}"
            ".legend .swatch{display:inline-block;width:16px;height:10px;border-radius:999px;}"
            ".high{background:rgba(217,48,37,0.2);}"
            ".low{background:rgba(15,139,44,0.15);}"
            ".unknown{background:rgba(85,98,142,0.2);}"
            ".synonym{background:rgba(66,135,245,0.25);}"
            ".segment{display:inline;}"
            ".segment.high{background:rgba(217,48,37,0.2);}"
            ".segment.low{background:rgba(15,139,44,0.15);}"
            ".segment.unknown{background:rgba(85,98,142,0.2);}"
            ".segment.synonym{background:rgba(66,135,245,0.25);}"
            ".sample{background:#fff;border:1px solid #e1e6f3;border-radius:12px;padding:18px;margin-bottom:20px;}"
            ".meta{font-size:13px;color:#5a688f;margin-bottom:12px;}"
            ".segments{font-family:monospace;white-space:pre-wrap;line-height:1.55;}"
            "</style></head><body>"
        )
        writer.write("<h1>Entropy Segments</h1>")
        writer.write(
            "<div class='legend'>"
            "<span><span class='swatch high'></span>High entropy</span>"
            "<span><span class='swatch low'></span>Low entropy</span>"
            "<span><span class='swatch synonym'></span>Synonym-like high entropy</span>"
            "<span><span class='swatch unknown'></span>Unknown / skipped</span>"
            "</div>"
        )
        for sample in samples:
            sample_id = html.escape(str(sample.get('id')))
            threshold_value = sample.get('threshold')
            source = sample.get('threshold_source')
            if isinstance(threshold_value, (int, float)):
                threshold_str = f"{threshold_value:.6f}"
            else:
                threshold_str = str(threshold_value)
            source_str = html.escape(str(source))
            writer.write("<section class='sample'>")
            writer.write(f"<h2>Sample {sample_id}</h2>")
            writer.write(f"<div class='meta'>Threshold: {threshold_str} "
                         f"(source: {source_str})</div>")
            writer.write("<div class='segments'>")
            for segment in sample.get('segments', []):
                seg_type = html.escape(segment.get('type', 'unknown'))
                seg_text = segment.get('text', '')
                seg_html = html.escape(seg_text)
                writer.write(f"<span class='segment {seg_type}'>{seg_html}</span>")
            writer.write("</div></section>")
        writer.write("</body></html>")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.filter_synonym_high and not WORDNET_AVAILABLE:
        print('Warning: NLTK wordnet not available; synonym filtering disabled.')
    process_observations(
        input_path=input_path,
        output_path=output_path,
        threshold=args.threshold,
        percentile=args.percentile,
        skip_unknown=args.skip_unknown,
        filter_synonym_high=args.filter_synonym_high and WORDNET_AVAILABLE,
    )


if __name__ == '__main__':
    main()
