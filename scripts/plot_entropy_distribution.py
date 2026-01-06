#!/usr/bin/env python3
"""
Plot token entropy distribution for each sample in an entropy observations file.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot token entropy distributions from observations.jsonl')
    parser.add_argument('--input', required=True, help='Path to observations.jsonl file')
    parser.add_argument('--output-dir', required=True, help='Directory to store generated PNG plots')
    parser.add_argument('--limit', type=int, default=None, help='Optional cap on number of samples to process')
    parser.add_argument('--skip-empty', action='store_true', help='Skip samples without token entropy data')
    return parser.parse_args()


def safe_name(name: Optional[str], fallback: str) -> str:
    if not name:
        return fallback
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(name)).strip('_')
    return slug or fallback


def sanitize_series(values: Iterable[Optional[float]]) -> List[float]:
    result: List[float] = []
    for value in values:
        if value is None:
            result.append(math.nan)
        else:
            result.append(float(value))
    return result


def derive_status(score: Optional[dict]) -> tuple[str, str]:
    default = ('Unknown', '#55628e')
    if not isinstance(score, dict):
        return default

    values = score.get('values')
    main_value = score.get('main_value')
    status_bool: Optional[bool] = None

    if isinstance(values, dict):
        for value in values.values():
            if isinstance(value, bool):
                status_bool = value
                break

    if status_bool is None:
        if isinstance(main_value, bool):
            status_bool = main_value
        elif isinstance(main_value, (int, float)) and math.isfinite(main_value):
            if math.isclose(main_value, 1.0):
                status_bool = True
            elif math.isclose(main_value, 0.0):
                status_bool = False

    if status_bool is True:
        return ('Correct', '#0f8b2c')
    if status_bool is False:
        return ('Incorrect', '#d93025')
    return default


def plot_entropy(record_id: str, entropies: List[float], status: tuple[str, str], output_path: Path) -> None:
    plt.figure(figsize=(12, 4))
    x_values = range(len(entropies))
    plt.plot(x_values, entropies, linewidth=1.0, color='#1f77b4')
    label, color = status
    plt.title(f'Token Entropy Distribution · Sample {record_id} · {label}', color=color)
    plt.xlabel('Token Index')
    plt.ylabel('Entropy')
    plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    with input_path.open('r', encoding='utf-8') as handle:
        for line_idx, line in enumerate(handle, start=1):
            if args.limit is not None and processed >= args.limit:
                break

            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            metrics = data.get('metrics') or {}
            entropies_raw = metrics.get('token_entropies')
            if entropies_raw is None:
                if args.skip_empty:
                    continue
                entropies = []
            else:
                entropies = sanitize_series(entropies_raw)

            if not entropies and args.skip_empty:
                continue

            record_id = safe_name(data.get('id'), f'sample_{line_idx:04d}')
            status = derive_status(data.get('score'))
            output_path = output_dir / f'{record_id}.png'

            plot_entropy(record_id, entropies, status, output_path)
            processed += 1

    print(f'Generated {processed} plot(s) in {output_dir}')


if __name__ == '__main__':
    main()
