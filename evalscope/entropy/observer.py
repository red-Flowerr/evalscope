from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from evalscope.api.evaluator.state import TaskState
from evalscope.api.model.model_output import Logprob
from evalscope.utils.io_utils import safe_filename
from evalscope.utils.logger import get_logger

if TYPE_CHECKING:
    from evalscope.api.metric.scorer import SampleScore

logger = get_logger()


@dataclass(slots=True)
class LLMToken:
    text: str
    logprob: float
    top_logprobs: Optional[Dict[str, float]] = None
    is_special: bool = False


@dataclass(slots=True)
class LLMGeneration:
    text: str
    tokens: List[LLMToken]


@dataclass(slots=True)
class Observation:
    record_id: str
    metrics: Dict[str, Any]
    response_text: str
    tokens: List[LLMToken]
    metadata: Dict[str, Any]
    score: Optional[Dict[str, Any]] = None
    token_categories: Optional[List[str]] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            'id': self.record_id,
            'response': self.response_text,
            'metrics': self.metrics,
            'metadata': self.metadata,
            'score': self.score,
            'token_categories': self.token_categories,
            'tokens': [
                {
                    'text': token.text,
                    'logprob': token.logprob,
                    'top_logprobs': token.top_logprobs,
                    'is_special': token.is_special,
                }
                for token in self.tokens
            ],
        }


def compute_entropy_metrics(generation: LLMGeneration) -> Dict[str, Any]:
    per_token_entropies: List[Optional[float]] = []
    numeric_entropies: List[float] = []
    for token in generation.tokens:
        if token.is_special:
            per_token_entropies.append(None)
            continue
        entropy = _token_entropy(token)
        if entropy is not None:
            numeric_entropies.append(entropy)
        per_token_entropies.append(entropy)

    if not numeric_entropies:
        return {
            'mean_token_entropy': None,
            'max_token_entropy': None,
            'min_token_entropy': None,
            'median_token_entropy': None,
            'std_token_entropy': None,
            'p75_token_entropy': None,
            'p90_token_entropy': None,
            'fraction_high_entropy_tokens': None,
            'fraction_zero_entropy_tokens': None,
            'token_entropies': per_token_entropies,
        }

    mean_entropy = sum(numeric_entropies) / len(numeric_entropies)
    max_entropy = max(numeric_entropies)
    min_entropy = min(numeric_entropies)
    median_entropy = statistics.median(numeric_entropies)
    std_entropy = statistics.pstdev(numeric_entropies) if len(numeric_entropies) > 1 else 0.0
    p75_entropy = _percentile(numeric_entropies, 0.75)
    p90_entropy = _percentile(numeric_entropies, 0.90)
    high_threshold = 1.0
    high_count = sum(1 for value in numeric_entropies if value >= high_threshold)
    zero_count = sum(1 for value in numeric_entropies if math.isclose(value, 0.0, abs_tol=1e-9))
    total = len(numeric_entropies)
    fraction_high = high_count / total if total else 0.0
    fraction_zero = zero_count / total if total else 0.0

    return {
        'mean_token_entropy': mean_entropy,
        'max_token_entropy': max_entropy,
        'min_token_entropy': min_entropy,
        'median_token_entropy': median_entropy,
        'std_token_entropy': std_entropy,
        'p75_token_entropy': p75_entropy,
        'p90_token_entropy': p90_entropy,
        'fraction_high_entropy_tokens': fraction_high,
        'fraction_zero_entropy_tokens': fraction_zero,
        'token_entropies': per_token_entropies,
    }


def _token_entropy(token: LLMToken) -> Optional[float]:
    candidates: Dict[str, float] = {}
    if token.top_logprobs:
        candidates.update(token.top_logprobs)
    candidates[token.text] = token.logprob

    probs = [math.exp(logprob) for logprob in candidates.values()]
    total = sum(probs)
    if total <= 0:
        return None

    normalized = [prob / total for prob in probs if prob > 0]
    if not normalized:
        return None

    return -sum(prob * math.log(prob) for prob in normalized)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return math.nan
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return sorted_values[int(idx)]
    fraction = idx - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _categorize_tokens(tokens: List[LLMToken]) -> tuple[List[str], float, int]:
    categories: List[str] = []
    code_mode = False
    code_count = 0
    total_non_special = 0

    for token in tokens:
        text = token.text or ''
        if token.is_special:
            categories.append('special')
            continue

        if '```' in text:
            categories.append('code-fence')
            code_mode = not code_mode
            continue

        category = 'code' if code_mode else 'text'
        categories.append(category)
        total_non_special += 1
        if category == 'code':
            code_count += 1

    fraction_code = code_count / total_non_special if total_non_special else 0.0
    return categories, fraction_code, code_count


class EntropyReportBuilder:
    def __init__(
        self,
        base_dir: Path,
        *,
        benchmark_name: str,
        include_prompt: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.benchmark_name = benchmark_name
        self.include_prompt = include_prompt
        self.max_samples = max_samples
        self._records: Dict[str, List[Observation]] = {}
        self._missing_logprobs = 0

    def add_task_states(self, subset: str, task_states: Sequence[TaskState]) -> None:
        if not task_states:
            return
        bucket = self._records.setdefault(subset, [])
        for state in task_states:
            observation = self._build_observation(state)
            if observation is None:
                self._missing_logprobs += 1
                continue
            bucket.append(observation)

    def update_scores(self, subset: str, sample_scores: Sequence['SampleScore']) -> None:
        """
        Attach review scores to existing observations so downstream HTML can surface correctness.
        """
        if not sample_scores:
            return
        observations = self._records.get(subset)
        if not observations:
            return

        lookup = {obs.record_id: obs for obs in observations}
        for sample_score in sample_scores:
            if sample_score is None or sample_score.score is None:
                continue

            record_id = str(sample_score.sample_id)
            observation = lookup.get(record_id)
            if observation is None:
                continue

            score_obj = sample_score.score
            try:
                main_value = score_obj.main_value
            except Exception:
                main_value = None

            observation.score = {
                'values': dict(score_obj.value),
                'main_name': score_obj.main_score_name,
                'main_value': main_value,
                'explanation': score_obj.explanation,
            }

    def finalize(self) -> Optional[Dict[str, Any]]:
        if not self._records:
            if self._missing_logprobs:
                logger.warning(
                    'Entropy observer skipped %d samples without logprobs for %s.',
                    self._missing_logprobs,
                    self.benchmark_name,
                )
            return None

        self.base_dir.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, Any] = {}
        for subset, observations in self._records.items():
            limited = observations
            if self.max_samples is not None and len(limited) > self.max_samples:
                limited = limited[: self.max_samples]
            subset_summary = self._write_subset(subset, limited)
            summary[subset] = subset_summary

        if self._missing_logprobs:
            summary['missing_logprobs'] = self._missing_logprobs
            logger.warning(
                'Entropy observer skipped %d samples without logprobs for %s.',
                self._missing_logprobs,
                self.benchmark_name,
            )

        summary_path = self.base_dir / 'summary.json'
        with summary_path.open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        logger.info('Saved entropy summary to %s', summary_path)
        return {'summary_path': str(summary_path), 'subsets': summary}

    def _write_subset(self, subset: str, observations: List[Observation]) -> Dict[str, Any]:
        safe_name = safe_filename(subset or 'default')
        subset_dir = self.base_dir / safe_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = subset_dir / 'observations.jsonl'
        index_html_path = subset_dir / 'index.html'

        with jsonl_path.open('w', encoding='utf-8') as handle:
            for observation in observations:
                handle.write(json.dumps(observation.to_json_dict(), ensure_ascii=False))
                handle.write('\n')

        stats = self._summaries(observations)
        html_files = self._write_observation_pages(subset_dir, subset, observations, stats)
        self._write_index_page(index_html_path, subset, stats, html_files)

        return {
            'count': len(observations),
            'stats': stats,
            'jsonl_path': str(jsonl_path),
            'html_index': str(index_html_path),
            'html_files': html_files,
        }

    def _summaries(self, observations: List[Observation]) -> Dict[str, Any]:
        def collect(key: str) -> List[float]:
            values: List[float] = []
            for obs in observations:
                value = obs.metrics.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(value)
            return values

        def summarize(values: List[float]) -> Optional[Dict[str, float]]:
            if not values:
                return None
            values_sorted = sorted(values)
            return {
                'mean': sum(values_sorted) / len(values_sorted),
                'min': values_sorted[0],
                'max': values_sorted[-1],
            }

        metric_keys = [
            'mean_token_entropy',
            'median_token_entropy',
            'max_token_entropy',
            'min_token_entropy',
            'std_token_entropy',
            'p75_token_entropy',
            'p90_token_entropy',
            'fraction_high_entropy_tokens',
            'fraction_zero_entropy_tokens',
            'fraction_code_tokens',
            'code_token_count',
        ]

        ranked = sorted(
            [
                (obs.record_id, obs.metrics.get('mean_token_entropy'))
                for obs in observations
                if obs.metrics.get('mean_token_entropy') is not None
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        summary: Dict[str, Any] = {key: summarize(collect(key)) for key in metric_keys}
        summary['highest_mean_entropy_samples'] = ranked[:5]
        return summary

    def _score_status(self, observation: Observation) -> Dict[str, Any]:
        info = observation.score or {}
        if not info:
            return {
                'badge_text': 'N/A',
                'badge_class': 'status-unknown',
                'status_bool': None,
                'main_name': None,
                'main_value': None,
                'values': {},
                'explanation': None,
            }

        values = info.get('values') or {}
        main_value = info.get('main_value')
        main_name = info.get('main_name') or info.get('main_score_name')

        status_bool = None
        bool_value = next((v for v in values.values() if isinstance(v, bool)), None)
        if bool_value is not None:
            status_bool = bool_value
        elif isinstance(main_value, bool):
            status_bool = main_value
        elif isinstance(main_value, (int, float)) and math.isfinite(main_value):
            if math.isclose(main_value, 1.0):
                status_bool = True
            elif math.isclose(main_value, 0.0):
                status_bool = False
            elif 0 <= main_value <= 1:
                status_bool = main_value >= 0.5

        badge_class = 'status-unknown'
        badge_text = 'N/A'
        if status_bool is True:
            badge_class = 'status-correct'
            badge_text = 'Correct'
        elif status_bool is False:
            badge_class = 'status-incorrect'
            badge_text = 'Incorrect'
        else:
            if isinstance(main_value, (int, float)) and math.isfinite(main_value):
                badge_text = f'{main_value:.4f}'

        return {
            'badge_text': badge_text,
            'badge_class': badge_class,
            'status_bool': status_bool,
            'main_name': main_name,
            'main_value': main_value,
            'values': values,
            'explanation': info.get('explanation'),
        }

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            if math.isnan(value):
                return 'nan'
            return f'{value:.4f}'
        return str(value)

    @staticmethod
    def _format_stat_summary(stat: Optional[Dict[str, float]]) -> str:
        if not stat:
            return 'N/A'
        mean = stat.get('mean')
        min_value = stat.get('min')
        max_value = stat.get('max')
        parts = []
        if mean is not None:
            parts.append(f'mean {mean:.4f}')
        if min_value is not None:
            parts.append(f'min {min_value:.4f}')
        if max_value is not None:
            parts.append(f'max {max_value:.4f}')
        return ' · '.join(parts) if parts else 'N/A'

    def _build_observation(self, task_state: TaskState) -> Optional[Observation]:
        output = task_state.output
        if output is None or output.empty:
            return None
        if not output.choices:
            return None
        choice = output.choices[0]
        if choice.logprobs is None or not choice.logprobs.content:
            return None

        tokens: List[LLMToken] = []
        for entry in choice.logprobs.content:
            top_dict = (
                {item.token or '': item.logprob for item in entry.top_logprobs}
                if entry.top_logprobs
                else {}
            )
            tokens.append(
                LLMToken(
                    text=entry.token or '',
                    logprob=entry.logprob,
                    top_logprobs=top_dict or None,
                    is_special=_is_special(entry),
                )
            )

        generation = LLMGeneration(
            text=choice.message.text or '',
            tokens=tokens,
        )
        metrics = compute_entropy_metrics(generation)
        token_categories, code_fraction, code_count = _categorize_tokens(tokens)
        metrics['fraction_code_tokens'] = code_fraction
        metrics['code_token_count'] = code_count
        target_obj = task_state.target
        if hasattr(target_obj, 'text'):
            target_text = target_obj.text
        elif isinstance(target_obj, str):
            target_text = target_obj
        else:
            target_text = None

        metadata: Dict[str, Any] = {
            'sample_id': task_state.sample_id,
            'group_id': task_state.group_id,
            'target': target_text,
            'stop_reason': choice.stop_reason,
            'usage': output.usage.model_dump() if output.usage else None,
            'sample_metadata': task_state.metadata,
        }
        if self.include_prompt:
            metadata['prompt'] = task_state.input_markdown

        return Observation(
            record_id=str(task_state.sample_id),
            metrics=metrics,
            response_text=generation.text,
            tokens=tokens,
            metadata=metadata,
            token_categories=token_categories,
        )

    def _write_observation_pages(
        self,
        subset_dir: Path,
        subset: str,
        observations: List[Observation],
        stats: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        html_files: List[Dict[str, str]] = []
        total = len(observations)
        for idx, observation in enumerate(observations, start=1):
            record_id_value = str(observation.record_id) if observation.record_id else f'sample_{idx}'
            slug = safe_filename(record_id_value) or f'sample_{idx}'
            filename = f'{idx:03d}_{slug}.html'
            html_path = subset_dir / filename
            status_info = self._score_status(observation)
            self._write_single_observation_html(
                html_path=html_path,
                subset=subset,
                observation=observation,
                position=idx,
                total=total,
                stats=stats,
                status_info=status_info,
            )
            html_files.append(
                {
                    'record_id': observation.record_id,
                    'html_path': str(html_path),
                    'status': status_info,
                }
            )
        return html_files

    def _write_index_page(
        self,
        index_path: Path,
        subset: str,
        stats: Dict[str, Any],
        html_files: List[Dict[str, str]],
    ) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        top_samples = stats.get('highest_mean_entropy_samples') or []

        with index_path.open('w', encoding='utf-8') as handle:
            handle.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
            handle.write(f"<title>Entropy Report Index - {escape(self.benchmark_name)}</title>")
            handle.write(
                "<style>"
                "body{font-family:system-ui,-apple-system,\"Segoe UI\",sans-serif;background:#f5f6fb;color:#1f2530;margin:0;padding:24px;}"
                "h1{margin-top:0;font-size:24px;color:#182140;margin-bottom:24px;}"
                ".summary{background:#fff;border:1px solid #dfe4f6;border-radius:10px;padding:18px;margin-bottom:24px;}"
                ".summary h2{margin:0 0 12px;font-size:16px;color:#273352;}"
                ".summary ul{list-style:none;padding:0;margin:0;display:grid;gap:6px;}"
                ".summary li{font-size:13px;color:#354167;}"
                ".list{background:#fff;border:1px solid #e1e6f3;border-radius:10px;padding:18px;}"
                ".list table{width:100%;border-collapse:collapse;}"
                ".list th,.list td{text-align:left;padding:10px;border-bottom:1px solid #edf1ff;font-size:13px;color:#31416f;}"
                ".list tr:hover{background:#f8faff;}"
                "a{color:#2a4bd7;text-decoration:none;}"
                "a:hover{text-decoration:underline;}"
                ".status-badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;padding:4px 10px;border-radius:999px;}"
                ".status-correct{background:#e7f6ed;color:#0f8b2c;}"
                ".status-incorrect{background:#fde7e7;color:#d93025;}"
                ".status-unknown{background:#eef1ff;color:#55628e;}"
                ".status-detail{font-size:11px;color:#5a688f;margin-top:4px;}"
                "</style></head><body>"
            )
            handle.write(
                f"<h1>Token Entropy Report · {escape(self.benchmark_name)} · {escape(subset or 'default')}</h1>"
            )
            handle.write("<section class='summary'><h2>Summary</h2><ul>")
            handle.write(f"<li>Total samples: {len(html_files)}</li>")
            summary_metrics = [
                ('mean_token_entropy', 'Mean token entropy'),
                ('median_token_entropy', 'Median token entropy'),
                ('max_token_entropy', 'Max token entropy'),
                ('min_token_entropy', 'Min token entropy'),
                ('std_token_entropy', 'Std token entropy'),
                ('p75_token_entropy', '75th percentile entropy'),
                ('p90_token_entropy', '90th percentile entropy'),
                ('fraction_high_entropy_tokens', 'Fraction of tokens ≥ 1.0 entropy'),
                ('fraction_zero_entropy_tokens', 'Fraction of zero-entropy tokens'),
                ('fraction_code_tokens', 'Fraction of code tokens'),
                ('code_token_count', 'Code token count'),
            ]
            for key, label in summary_metrics:
                stat = stats.get(key)
                handle.write(f"<li>{label}: {self._format_stat_summary(stat)}</li>")
            if top_samples:
                formatted = ', '.join(
                    f"{escape(sample_id)} ({value:.4f})"
                    for sample_id, value in top_samples
                    if value is not None
                )
                if formatted:
                    handle.write(f"<li>Highest mean entropy: {formatted}</li>")
                else:
                    handle.write("<li>Highest mean entropy: N/A</li>")
            handle.write("</ul></section>")

            handle.write("<section class='list'><h2>Samples</h2><table>")
            handle.write("<tr><th>#</th><th>Record ID</th><th>Result</th><th>Report</th></tr>")
            for idx, item in enumerate(html_files, start=1):
                record_id = item['record_id'] or f'Sample {idx}'
                rel_path = Path(item['html_path']).name
                status_info = item.get('status') or {}
                badge_text = escape(str(status_info.get('badge_text', 'N/A')))
                badge_class = status_info.get('badge_class', 'status-unknown')
                main_name = status_info.get('main_name')
                main_value = status_info.get('main_value')
                detail = ''
                if main_name and main_value is not None:
                    detail_value = escape(self._format_metric_value(main_value))
                    detail = f"<div class='status-detail'>{escape(str(main_name))}: {detail_value}</div>"
                handle.write(
                    f"<tr><td>{idx}</td><td>{escape(str(record_id))}</td>"
                    f"<td><span class='status-badge {badge_class}'>{badge_text}</span>{detail}</td>"
                    f"<td><a href='{escape(rel_path)}' target='_blank'>Open report</a></td></tr>"
                )
            handle.write("</table></section>")
            handle.write("</body></html>")

    def _write_single_observation_html(
        self,
        *,
        html_path: Path,
        subset: str,
        observation: Observation,
        position: int,
        total: int,
        stats: Dict[str, Any],
        status_info: Dict[str, Any],
    ) -> None:
        html_path.parent.mkdir(parents=True, exist_ok=True)

        entropies: List[Optional[float]] = observation.metrics.get('token_entropies') or []
        max_entropy = observation.metrics.get('max_token_entropy')

        display_tokens: List[Dict[str, Any]] = []
        row_index = 0
        for idx, token in enumerate(observation.tokens):
            category = 'text'
            if observation.token_categories and idx < len(observation.token_categories):
                category = observation.token_categories[idx] or 'text'
            if token.is_special or category == 'special':
                continue
            row_index += 1
            entropy_value = entropies[idx] if idx < len(entropies) else None
            if entropy_value is not None and abs(entropy_value) < 1e-9:
                entropy_value = 0.0
            top_logprobs = token.top_logprobs or {}
            if max_entropy and max_entropy > 0 and entropy_value is not None:
                normalized = max(0.0, min(1.0, entropy_value / max_entropy))
            else:
                normalized = 0.0
            highlight_alpha = 0.20 + (0.70 * normalized)
            highlight_alpha = max(0.0, min(0.95, highlight_alpha))
            entropy_attr = f"{entropy_value:.4f}" if entropy_value is not None else "nan"
            if entropy_value is not None:
                chip_style = f"background-color: rgba(255, 99, 71, {highlight_alpha:.3f});color:#241517;"
            else:
                chip_style = "background-color: #e9ecf5;"
            logprob_value = token.logprob
            top_items = list(top_logprobs.items())
            tooltip_rows_html = "".join(
                f"<div class='tooltip-logprob-row'><span class='candidate'>{escape(_format_token_text(tok))}</span>"
                f"<span class='score'>{value:.4f}</span></div>"
                for tok, value in top_items
            )
            if entropy_value is not None:
                entropy_display = f"{entropy_value:.4f}"
            else:
                entropy_display = "N/A"
            if logprob_value is not None:
                logprob_display = (
                    f"{0.0:.4f}" if abs(logprob_value) < 1e-9 else f"{logprob_value:.4f}"
                )
            else:
                logprob_display = "N/A"
            tooltip_logprobs_block = (
                "<div class='tooltip-logprobs-title'>Top logprobs</div>"
                f"<div class='tooltip-logprobs'>{tooltip_rows_html}</div>"
                if tooltip_rows_html
                else "<div class='tooltip-logprobs-title'>Top logprobs</div><div class='tooltip-empty'>No data</div>"
            )
            tooltip_html = (
                "<div class='tooltip'>"
                f"<div class='tooltip-entropy'>Entropy: {entropy_display}</div>"
                f"<div class='tooltip-logprob'>Logprob: {logprob_display}</div>"
                f"{tooltip_logprobs_block}"
                "</div>"
            )
            title_parts = [
                f"Token #{row_index} (orig {idx})",
                f"logprob: {logprob_display}",
                f"entropy: {entropy_display}",
            ]
            title_attr = " | ".join(title_parts)
            display_tokens.append(
                {
                    'response_text': escape(_render_token_text(token.text)),
                    'entropy_attr': entropy_attr,
                    'is_whitespace': token.text.strip() == '',
                    'chip_style': chip_style,
                    'tooltip_html': tooltip_html,
                    'title_attr': title_attr,
                    'category': category,
                }
            )

        with html_path.open('w', encoding='utf-8') as handle:
            handle.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
            handle.write(
                f"<title>Entropy Report - {escape(self.benchmark_name)} - {escape(str(observation.record_id))}</title>"
            )
            handle.write(
                "<style>"
                "body{font-family:system-ui,-apple-system,\"Segoe UI\",sans-serif;background:#f5f6fb;color:#1f2530;margin:0;padding:24px;}"
                "h1{margin-top:0;font-size:22px;color:#182140;margin-bottom:20px;}"
                ".breadcrumb{font-size:12px;color:#53608f;margin-bottom:16px;}"
                ".metric-line{font-size:13px;color:#3b4b7a;margin-bottom:12px;}"
                ".prompt-block{margin-bottom:16px;padding:14px 16px;border:1px solid #dfe4f6;border-radius:8px;background:#f8faff;}"
                ".prompt-label{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#4a5c91;margin-bottom:8px;display:block;}"
                ".prompt-text{white-space:pre-wrap;font-family:monospace;font-size:13px;line-height:1.5;color:#1f2d4d;}"
                ".response-stream{white-space:pre-wrap;font-family:monospace;font-size:13px;line-height:1.6;background:#fff;border:1px solid #e3e8f8;border-radius:8px;padding:16px;color:#19213b;word-break:break-word;overflow-wrap:anywhere;}"
                ".response-token{display:inline-block;white-space:pre-wrap;border-radius:4px;padding:2px 3px;margin:0 1px;transition:box-shadow 0.1s ease,transform 0.1s ease;color:inherit;position:relative;word-break:break-word;overflow-wrap:anywhere;}"
                ".response-token:hover{box-shadow:0 4px 12px rgba(31,37,48,0.18);transform:translateY(-1px);z-index:2;}"
                ".response-token.whitespace{margin:0;}"
                ".response-token .tooltip{position:absolute;left:0;bottom:100%;transform:translateY(-6px);background:#1f2530;color:#f8f9ff;padding:8px 10px;border-radius:6px;font-size:12px;line-height:1.45;white-space:normal;box-shadow:0 8px 18px rgba(15,23,42,0.3);opacity:0;visibility:hidden;pointer-events:none;min-width:220px;max-width:320px;}"
                ".response-token:hover .tooltip{opacity:1;visibility:visible;}"
                ".response-token .tooltip::after{content:\"\";position:absolute;top:100%;left:12px;border-width:6px;border-style:solid;border-color:#1f2530 transparent transparent transparent;}"
                ".tooltip-entropy{font-weight:600;margin-bottom:4px;color:#fefefe;}"
                ".tooltip-logprob{font-family:monospace;font-size:12px;margin-bottom:6px;color:rgba(248,249,255,0.9);}"
                ".tooltip-logprobs-title{font-size:11px;text-transform:uppercase;letter-spacing:0.05em;color:rgba(248,249,255,0.7);margin-bottom:4px;}"
                ".tooltip-logprobs{display:flex;flex-direction:column;gap:2px;}"
                ".tooltip-logprob-row{display:flex;justify-content:space-between;gap:12px;font-family:monospace;font-size:12px;color:#fefefe;}"
                ".tooltip-empty{font-style:italic;color:rgba(248,249,255,0.7);}"
                ".status-badge{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;padding:4px 10px;border-radius:999px;}"
                ".status-correct{background:#e7f6ed;color:#0f8b2c;}"
                ".status-incorrect{background:#fde7e7;color:#d93025;}"
                ".status-unknown{background:#eef1ff;color:#55628e;}"
                ".status-detail{font-size:12px;color:#5a688f;}"
                ".score-summary{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin-bottom:18px;padding:16px;border:1px solid #dfe4f6;border-radius:10px;background:#fff;}"
                ".score-metrics{list-style:none;padding:0;margin:0;display:flex;flex-wrap:wrap;gap:8px;font-size:12px;color:#3b4b7a;}"
                ".score-metrics li{background:#eef1ff;border-radius:6px;padding:4px 8px;}"
                ".score-metrics .metric-name{font-weight:600;margin-right:6px;}"
                ".score-explanation{font-size:12px;color:#5a688f;margin-top:8px;white-space:pre-wrap;}"
                ".response-token.code{box-shadow:inset 0 -2px 0 rgba(15,139,44,0.45);}"
                ".response-token.code-fence{background:#fff3cd;color:#8a6d3b;font-weight:600;}"
                ".response-token.special{opacity:0.6;}"
                ".legend{font-size:11px;color:#5a688f;margin-bottom:12px;display:flex;gap:12px;flex-wrap:wrap;align-items:center;}"
                ".legend-item{display:inline-flex;align-items:center;gap:6px;padding:4px 6px;border-radius:6px;background:#eef1ff;}"
                ".legend-swatch{width:14px;height:6px;border-radius:999px;display:inline-block;}"
                ".legend-code{background:rgba(15,139,44,0.6);}"
                ".legend-text{background:rgba(49,65,111,0.45);}"
                ".legend-fence{background:#fff3cd;border:1px solid #f0c36d;}"
                "</style></head><body>"
            )
            handle.write(
                f"<div class='breadcrumb'><a href='index.html'>⟵ Back to index</a> · Sample {position} / {total}</div>"
            )
            handle.write(
                f"<h1>{escape(self.benchmark_name)} · {escape(subset or 'default')} · Sample {escape(str(observation.record_id))}</h1>"
            )

            badge_text = escape(str(status_info.get('badge_text', 'N/A')))
            badge_class = status_info.get('badge_class', 'status-unknown')
            main_name = status_info.get('main_name')
            main_value = status_info.get('main_value')
            explanation = status_info.get('explanation')
            values_dict = status_info.get('values') or {}

            if main_name and main_value is not None:
                detail_text = f"<div class='status-detail'>{escape(str(main_name))}: {escape(self._format_metric_value(main_value))}</div>"
            else:
                detail_text = ""

            metrics_list_html = "".join(
                f"<li><span class='metric-name'>{escape(str(k))}</span>"
                f"<span class='metric-value'>{escape(self._format_metric_value(v))}</span></li>"
                for k, v in values_dict.items()
            )
            explanation_html = (
                f"<div class='score-explanation'>{escape(str(explanation))}</div>" if explanation else ""
            )

            handle.write("<section class='score-summary'>")
            handle.write(f"<span class='status-badge {badge_class}'>{badge_text}</span>")
            if detail_text:
                handle.write(detail_text)
            if metrics_list_html:
                handle.write(f"<ul class='score-metrics'>{metrics_list_html}</ul>")
            if explanation_html:
                handle.write(explanation_html)
            handle.write("</section>")
            handle.write(
                "<div class='legend'>"
                "<span class='legend-item'><span class='legend-swatch legend-text'></span>Text token</span>"
                "<span class='legend-item'><span class='legend-swatch legend-code'></span>Code token</span>"
                "<span class='legend-item'><span class='legend-swatch legend-fence'></span>Code fence</span>"
                "</div>"
            )

            metric_line_parts = []
            per_sample_metrics = {
                'Mean': observation.metrics.get('mean_token_entropy'),
                'Median': observation.metrics.get('median_token_entropy'),
                'Std': observation.metrics.get('std_token_entropy'),
                'Min': observation.metrics.get('min_token_entropy'),
                'Max': observation.metrics.get('max_token_entropy'),
                'P75': observation.metrics.get('p75_token_entropy'),
                'P90': observation.metrics.get('p90_token_entropy'),
                'Frac ≥1.0': observation.metrics.get('fraction_high_entropy_tokens'),
                'Frac =0': observation.metrics.get('fraction_zero_entropy_tokens'),
                'Code frac': observation.metrics.get('fraction_code_tokens'),
            }
            for label, value in per_sample_metrics.items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metric_line_parts.append(f"{label}: {value:.4f}")
            code_token_count = observation.metrics.get('code_token_count')
            if isinstance(code_token_count, (int, float)) and code_token_count:
                metric_line_parts.append(f"Code tokens: {int(code_token_count)}")
            metric_line = " · ".join(metric_line_parts) if metric_line_parts else "Entropy metrics unavailable."
            handle.write(f"<div class='metric-line'>{metric_line}</div>")

            prompt_text = observation.metadata.get('prompt')
            if prompt_text:
                handle.write("<div class='prompt-block'>")
                handle.write("<span class='prompt-label'>Prompt</span>")
                handle.write(f"<div class='prompt-text'>{escape(str(prompt_text))}</div>")
                handle.write("</div>")

            handle.write("<div class='response-stream'>")
            for token_info in display_tokens:
                response_classes = ["response-token"]
                if token_info['is_whitespace']:
                    response_classes.append("whitespace")
                category = token_info.get('category', 'text')
                if category == 'code':
                    response_classes.append('code')
                elif category == 'code-fence':
                    response_classes.append('code-fence')
                elif category == 'special':
                    response_classes.append('special')
                handle.write(
                    f"<span class='{' '.join(response_classes)}' data-entropy='{token_info['entropy_attr']}' "
                    f"data-category='{category}' "
                    f"style='{token_info['chip_style']}' title='{escape(token_info['title_attr'])}'>"
                    f"{token_info['response_text']}{token_info['tooltip_html']}</span>"
                )
            handle.write("</div>")
            handle.write("</body></html>")


def _is_special(logprob: Logprob) -> bool:
    token = logprob.token or ''
    if token.startswith('<|') and token.endswith('|>'):
        return True
    if token.strip() == '' and not logprob.bytes:
        return False
    return False


def _format_token_text(text: str) -> str:
    return _render_token_text(text)


def _render_token_text(text: str) -> str:
    buffer: List[str] = []
    for ch in text:
        if ch == '\n':
            buffer.append('\\n')
        elif ch == '\r':
            buffer.append('\\r')
        elif ch == '\t':
            buffer.append('\\t')
        elif ch == '\u2028':
            buffer.append('\\u2028')
        elif ch == '\u2029':
            buffer.append('\\u2029')
        elif ch == 'Ġ':
            buffer.append(' ')
        elif ch.isprintable():
            buffer.append(ch)
        else:
            buffer.append(f'\\u{ord(ch):04x}')
    return ''.join(buffer)
