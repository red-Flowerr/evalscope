from __future__ import annotations

import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evalscope.api.evaluator.state import TaskState
from evalscope.api.model.model_output import Logprob
from evalscope.utils.io_utils import safe_filename
from evalscope.utils.logger import get_logger

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

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            'id': self.record_id,
            'response': self.response_text,
            'metrics': self.metrics,
            'metadata': self.metadata,
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
            'token_entropies': per_token_entropies,
        }

    mean_entropy = sum(numeric_entropies) / len(numeric_entropies)
    return {
        'mean_token_entropy': mean_entropy,
        'max_token_entropy': max(numeric_entropies),
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
                if value is not None:
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

        ranked = sorted(
            [
                (obs.record_id, obs.metrics.get('mean_token_entropy'))
                for obs in observations
                if obs.metrics.get('mean_token_entropy') is not None
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        return {
            'mean_token_entropy': summarize(collect('mean_token_entropy')),
            'max_token_entropy': summarize(collect('max_token_entropy')),
            'highest_mean_entropy_samples': ranked[:5],
        }

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
            self._write_single_observation_html(
                html_path=html_path,
                subset=subset,
                observation=observation,
                position=idx,
                total=total,
                stats=stats,
            )
            html_files.append({'record_id': observation.record_id, 'html_path': str(html_path)})
        return html_files

    def _write_index_page(
        self,
        index_path: Path,
        subset: str,
        stats: Dict[str, Any],
        html_files: List[Dict[str, str]],
    ) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        mean_stats = stats.get('mean_token_entropy')
        max_stats = stats.get('max_token_entropy')
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
                "</style></head><body>"
            )
            handle.write(
                f"<h1>Token Entropy Report · {escape(self.benchmark_name)} · {escape(subset or 'default')}</h1>"
            )
            handle.write("<section class='summary'><h2>Summary</h2><ul>")
            handle.write(f"<li>Total samples: {len(html_files)}</li>")
            handle.write(
                f"<li>Mean token entropy avg: {mean_stats['mean']:.4f}</li>"
                if mean_stats else "<li>Mean token entropy: N/A</li>"
            )
            handle.write(
                f"<li>Max token entropy avg: {max_stats['mean']:.4f}</li>"
                if max_stats else "<li>Max token entropy: N/A</li>"
            )
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
            handle.write("<tr><th>#</th><th>Record ID</th><th>Report</th></tr>")
            for idx, item in enumerate(html_files, start=1):
                record_id = item['record_id'] or f'Sample {idx}'
                rel_path = Path(item['html_path']).name
                handle.write(
                    f"<tr><td>{idx}</td><td>{escape(str(record_id))}</td>"
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
    ) -> None:
        html_path.parent.mkdir(parents=True, exist_ok=True)

        entropies: List[Optional[float]] = observation.metrics.get('token_entropies') or []
        max_entropy = observation.metrics.get('max_token_entropy')

        display_tokens: List[Dict[str, Any]] = []
        row_index = 0
        for idx, token in enumerate(observation.tokens):
            if token.is_special:
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
                ".response-stream{white-space:pre-wrap;font-family:monospace;font-size:13px;line-height:1.6;background:#fff;border:1px solid #e3e8f8;border-radius:8px;padding:16px;color:#19213b;}"
                ".response-token{display:inline;white-space:pre;border-radius:4px;padding:2px 3px;margin:0 1px;transition:box-shadow 0.1s ease,transform 0.1s ease;color:inherit;position:relative;}"
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
                "</style></head><body>"
            )
            handle.write(
                f"<div class='breadcrumb'><a href='index.html'>⟵ Back to index</a> · Sample {position} / {total}</div>"
            )
            handle.write(
                f"<h1>{escape(self.benchmark_name)} · {escape(subset or 'default')} · Sample {escape(str(observation.record_id))}</h1>"
            )

            mean_entropy = observation.metrics.get('mean_token_entropy')
            max_entropy_value = observation.metrics.get('max_token_entropy')
            if mean_entropy is not None and max_entropy_value is not None:
                metric_line = (
                    f"Mean token entropy: {mean_entropy:.4f} · "
                    f"Max token entropy: {max_entropy_value:.4f}"
                )
            else:
                metric_line = "Mean token entropy: N/A · Max token entropy: N/A"
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
                handle.write(
                    f"<span class='{' '.join(response_classes)}' data-entropy='{token_info['entropy_attr']}' "
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
