from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import statistics
from typing import Iterable


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _safe_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = float(len(values))
    xs = list(range(len(values)))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 1e-6:
        return 0.0
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values, strict=False))
    return float(numer / denom)


def _fmt_num(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_signed(value: float, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


@dataclass(frozen=True)
class AUSignal:
    presence: dict[str, float] = field(default_factory=dict)
    intensity: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BlinkSignal:
    blink_count: int = 0
    blink_rate_per_min: float = 0.0
    mean_inter_blink_interval: float = 0.0
    blink_rate_delta_vs_prev_window: float = 0.0
    blink_rate_delta_vs_rolling_baseline: float = 0.0
    blink_variability: float = 0.0


@dataclass(frozen=True)
class RMESSpotEvent:
    window_id: int
    clip_start_ts: float
    clip_end_ts: float
    peak_ts: float
    spot_score: float
    confidence: float
    quality_gate_passed: bool
    source: str = "rmes"
    interpretation: str = ""
    valence_delta: float = 0.0
    arousal_delta: float = 0.0
    dominant_emotion: str = "unknown"


@dataclass(frozen=True)
class WindowSample:
    ts: float
    valence: float
    arousal: float
    confidence: float
    frontal_score: float
    dominant_emotion: str
    face_scale: float = 0.0
    face_scale_bucket: str = "unknown"
    au_intensity: dict[str, float] = field(default_factory=dict)
    subtle_cues: list[dict[str, float | str]] = field(default_factory=list)
    blink_event: bool = False
    uncertain: bool = False


@dataclass(frozen=True)
class WindowStats:
    window_id: int
    mode: str
    reason: str
    start_ts: float
    end_ts: float
    sample_count: int
    duration_s: float
    valence_mean: float
    valence_min: float
    valence_max: float
    valence_std: float
    valence_slope: float
    arousal_mean: float
    arousal_min: float
    arousal_max: float
    arousal_std: float
    arousal_slope: float
    confidence_mean: float
    confidence_std: float
    frontal_score_mean: float
    frontal_score_std: float
    face_scale_mean: float = 0.0
    face_scale_std: float = 0.0
    dominant_scale_bucket: str = "unknown"
    dominant_affect_distribution: dict[str, int] = field(default_factory=dict)
    top_aus: list[tuple[str, float]] = field(default_factory=list)
    subtle_top: list[tuple[str, float]] = field(default_factory=list)
    blink_signal: BlinkSignal = field(default_factory=BlinkSignal)
    micro_expression_events: list[RMESSpotEvent] = field(default_factory=list)
    accepted_micro_expression_events: list[RMESSpotEvent] = field(default_factory=list)


@dataclass(frozen=True)
class RollingWindowSummary:
    window_count: int
    valence_mean: float
    valence_slope: float
    valence_volatility: float
    arousal_mean: float
    arousal_slope: float
    arousal_volatility: float
    confidence_mean: float
    face_scale_mean: float
    face_scale_slope: float
    blink_rate_mean: float
    blink_rate_delta_current_vs_baseline: float
    current_window_delta_vs_prev: dict[str, float] = field(default_factory=dict)
    top_aus: list[tuple[str, float]] = field(default_factory=list)
    subtle_top: list[tuple[str, float]] = field(default_factory=list)
    accepted_micro_expression_events: list[RMESSpotEvent] = field(default_factory=list)


@dataclass(frozen=True)
class StructuredContextSnapshot:
    backend: str
    backend_description: str
    current_window: WindowStats | None
    rolling_summary: RollingWindowSummary | None
    accepted_micro_expression_events: list[RMESSpotEvent] = field(default_factory=list)
    experimental_micro_expression_events: list[RMESSpotEvent] = field(default_factory=list)
    subtle_cues: list[dict[str, float | str]] = field(default_factory=list)
    pose_quality: dict[str, float] = field(default_factory=dict)
    face_scale: float = 0.0
    face_scale_bucket: str = "unknown"
    accepted_for_mainline: bool = False


FIELD_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "valence_mean": ("情绪效价均值", "窗口内正负向平均值，范围约 [-1, 1]，越大越偏正向。"),
    "arousal_mean": ("唤醒度均值", "窗口内整体激活程度，越高表示越紧绷或越激活。"),
    "confidence_mean": ("可靠度均值", "主视觉后端在窗口内的平均可信度。"),
    "frontal_score_mean": ("正视质量均值", "脸部面对镜头的稳定程度，越高越适合解释。"),
    "face_scale_mean": ("尺度均值", "人脸在画面中的占比均值，越大说明脸越近。"),
}


def compute_window_stats(
    *,
    window_id: int,
    mode: str,
    reason: str,
    start_ts: float,
    end_ts: float,
    samples: Iterable[WindowSample],
    blink_times: Iterable[float],
    experimental_events: Iterable[RMESSpotEvent],
    accepted_events: Iterable[RMESSpotEvent],
) -> WindowStats:
    sample_list = list(samples)
    blink_list = list(blink_times)
    experimental = list(experimental_events)
    accepted = list(accepted_events)
    duration_s = max(0.0, float(end_ts - start_ts))

    if not sample_list:
        return WindowStats(
            window_id=window_id,
            mode=mode,
            reason=reason,
            start_ts=start_ts,
            end_ts=end_ts,
            sample_count=0,
            duration_s=duration_s,
            valence_mean=0.0,
            valence_min=0.0,
            valence_max=0.0,
            valence_std=0.0,
            valence_slope=0.0,
            arousal_mean=0.0,
            arousal_min=0.0,
            arousal_max=0.0,
            arousal_std=0.0,
            arousal_slope=0.0,
            confidence_mean=0.0,
            confidence_std=0.0,
            frontal_score_mean=0.0,
            frontal_score_std=0.0,
            face_scale_mean=0.0,
            face_scale_std=0.0,
            dominant_scale_bucket="unknown",
            blink_signal=BlinkSignal(),
            micro_expression_events=experimental,
            accepted_micro_expression_events=accepted,
        )

    valences = [float(s.valence) for s in sample_list]
    arousals = [float(s.arousal) for s in sample_list]
    confidences = [float(s.confidence) for s in sample_list]
    frontals = [float(s.frontal_score) for s in sample_list]
    face_scales = [float(s.face_scale) for s in sample_list]
    dist = Counter(str(s.dominant_emotion) for s in sample_list)
    scale_bucket_dist = Counter(str(s.face_scale_bucket) for s in sample_list)
    au_acc: dict[str, list[float]] = {}
    subtle_acc: dict[str, float] = {}
    for sample in sample_list:
        for key, value in sample.au_intensity.items():
            au_acc.setdefault(str(key), []).append(float(value))
        for cue in sample.subtle_cues:
            name = str(cue.get("name", ""))
            subtle_acc[name] = subtle_acc.get(name, 0.0) + float(cue.get("strength", 0.0))
    top_aus = sorted(((k, _safe_mean(v)) for k, v in au_acc.items()), key=lambda item: item[1], reverse=True)[:5]
    subtle_top = sorted(subtle_acc.items(), key=lambda item: item[1], reverse=True)[:5]

    blink_count = len(blink_list)
    ibi_values = [max(0.0, b - a) for a, b in zip(blink_list[:-1], blink_list[1:], strict=False)]
    blink_rate = float(blink_count) * 60.0 / max(1.0, duration_s) if duration_s > 0 else 0.0
    blink_signal = BlinkSignal(
        blink_count=blink_count,
        blink_rate_per_min=blink_rate,
        mean_inter_blink_interval=_safe_mean(ibi_values),
        blink_variability=_safe_std(ibi_values),
    )

    return WindowStats(
        window_id=window_id,
        mode=mode,
        reason=reason,
        start_ts=start_ts,
        end_ts=end_ts,
        sample_count=len(sample_list),
        duration_s=duration_s,
        valence_mean=_safe_mean(valences),
        valence_min=min(valences),
        valence_max=max(valences),
        valence_std=_safe_std(valences),
        valence_slope=_safe_slope(valences),
        arousal_mean=_safe_mean(arousals),
        arousal_min=min(arousals),
        arousal_max=max(arousals),
        arousal_std=_safe_std(arousals),
        arousal_slope=_safe_slope(arousals),
        confidence_mean=_safe_mean(confidences),
        confidence_std=_safe_std(confidences),
        frontal_score_mean=_safe_mean(frontals),
        frontal_score_std=_safe_std(frontals),
        face_scale_mean=_safe_mean(face_scales),
        face_scale_std=_safe_std(face_scales),
        dominant_scale_bucket=scale_bucket_dist.most_common(1)[0][0] if scale_bucket_dist else "unknown",
        dominant_affect_distribution=dict(dist),
        top_aus=top_aus,
        subtle_top=subtle_top,
        blink_signal=blink_signal,
        micro_expression_events=experimental,
        accepted_micro_expression_events=accepted,
    )


def compute_rolling_summary(windows: list[WindowStats], *, max_windows: int) -> RollingWindowSummary | None:
    if not windows:
        return None
    selected = windows[-max_windows:]
    valence_means = [w.valence_mean for w in selected]
    arousal_means = [w.arousal_mean for w in selected]
    confidence_means = [w.confidence_mean for w in selected]
    face_scale_means = [w.face_scale_mean for w in selected]
    blink_rates = [w.blink_signal.blink_rate_per_min for w in selected]
    top_au_acc: dict[str, list[float]] = {}
    subtle_acc: dict[str, list[float]] = {}
    accepted_events: list[RMESSpotEvent] = []
    for window in selected:
        accepted_events.extend(window.accepted_micro_expression_events)
        for name, value in window.top_aus:
            top_au_acc.setdefault(name, []).append(float(value))
        for name, value in window.subtle_top:
            subtle_acc.setdefault(name, []).append(float(value))
    current = selected[-1]
    prev = selected[-2] if len(selected) >= 2 else None
    current_vs_prev = {
        "valence_mean_delta": float(current.valence_mean - prev.valence_mean) if prev else 0.0,
        "arousal_mean_delta": float(current.arousal_mean - prev.arousal_mean) if prev else 0.0,
        "confidence_mean_delta": float(current.confidence_mean - prev.confidence_mean) if prev else 0.0,
        "blink_rate_delta": float(current.blink_signal.blink_rate_per_min - prev.blink_signal.blink_rate_per_min) if prev else 0.0,
    }
    baseline = selected[:-1]
    baseline_blink = _safe_mean([w.blink_signal.blink_rate_per_min for w in baseline]) if baseline else 0.0
    return RollingWindowSummary(
        window_count=len(selected),
        valence_mean=_safe_mean(valence_means),
        valence_slope=_safe_slope(valence_means),
        valence_volatility=_safe_std(valence_means),
        arousal_mean=_safe_mean(arousal_means),
        arousal_slope=_safe_slope(arousal_means),
        arousal_volatility=_safe_std(arousal_means),
        confidence_mean=_safe_mean(confidence_means),
        face_scale_mean=_safe_mean(face_scale_means),
        face_scale_slope=_safe_slope(face_scale_means),
        blink_rate_mean=_safe_mean(blink_rates),
        blink_rate_delta_current_vs_baseline=float(current.blink_signal.blink_rate_per_min - baseline_blink),
        current_window_delta_vs_prev=current_vs_prev,
        top_aus=sorted(((k, _safe_mean(v)) for k, v in top_au_acc.items()), key=lambda item: item[1], reverse=True)[:5],
        subtle_top=sorted(((k, _safe_mean(v)) for k, v in subtle_acc.items()), key=lambda item: item[1], reverse=True)[:5],
        accepted_micro_expression_events=accepted_events[-6:],
    )


def format_window_stats_cn(window: WindowStats | None) -> str:
    if window is None:
        return "当前窗口：暂无数据。"
    lines = [
        f"当前窗口 #{window.window_id}",
        f"模式：{window.mode}；起因：{window.reason}；时长：{_fmt_num(window.duration_s, 1)} 秒；样本数：{window.sample_count}",
        f"- 情绪效价均值(valence_mean)：{_fmt_signed(window.valence_mean)}；说明：{FIELD_DESCRIPTIONS['valence_mean'][1]}",
        f"- 情绪效价范围：min={_fmt_signed(window.valence_min)} / max={_fmt_signed(window.valence_max)} / std={_fmt_num(window.valence_std)} / slope={_fmt_signed(window.valence_slope)}",
        f"- 唤醒度均值(arousal_mean)：{_fmt_num(window.arousal_mean)}；说明：{FIELD_DESCRIPTIONS['arousal_mean'][1]}",
        f"- 唤醒度范围：min={_fmt_num(window.arousal_min)} / max={_fmt_num(window.arousal_max)} / std={_fmt_num(window.arousal_std)} / slope={_fmt_signed(window.arousal_slope)}",
        f"- 可靠度均值(confidence_mean)：{_fmt_num(window.confidence_mean)}；说明：{FIELD_DESCRIPTIONS['confidence_mean'][1]}",
        f"- 正视质量均值(frontal_score_mean)：{_fmt_num(window.frontal_score_mean)}；说明：{FIELD_DESCRIPTIONS['frontal_score_mean'][1]}",
        f"- 尺度均值(face_scale_mean)：{_fmt_num(window.face_scale_mean)}；尺度桶：{window.dominant_scale_bucket}；说明：{FIELD_DESCRIPTIONS['face_scale_mean'][1]}",
        f"- 眨眼统计：count={window.blink_signal.blink_count} / rate={_fmt_num(window.blink_signal.blink_rate_per_min, 2)} 次/分钟 / ibi={_fmt_num(window.blink_signal.mean_inter_blink_interval, 2)} 秒 / variability={_fmt_num(window.blink_signal.blink_variability, 2)}",
        f"- 主导情绪分布：{window.dominant_affect_distribution or {'unknown': 0}}",
        f"- 主要 AU：{window.top_aus or []}",
        f"- 主要细粒度线索：{window.subtle_top or []}",
    ]
    if window.accepted_micro_expression_events:
        lines.append(f"- 已接纳微表情事件：{len(window.accepted_micro_expression_events)} 个")
    elif window.micro_expression_events:
        lines.append(f"- 实验微表情事件：{len(window.micro_expression_events)} 个（尚未接纳到主线）")
    return "\n".join(lines)


def format_rolling_summary_cn(summary: RollingWindowSummary | None) -> str:
    if summary is None:
        return "最近窗口摘要：暂无数据。"
    lines = [
        f"最近 {summary.window_count} 个窗口摘要",
        f"- valence_mean：{_fmt_signed(summary.valence_mean)}；说明：最近窗口的平均正负向中心。",
        f"- valence_slope：{_fmt_signed(summary.valence_slope)}；说明：最近窗口间正负向变化趋势。",
        f"- valence_volatility：{_fmt_num(summary.valence_volatility)}；说明：最近窗口均值波动。",
        f"- arousal_mean：{_fmt_num(summary.arousal_mean)}；说明：最近窗口平均激活程度。",
        f"- arousal_slope：{_fmt_signed(summary.arousal_slope)}；说明：最近窗口激活趋势。",
        f"- arousal_volatility：{_fmt_num(summary.arousal_volatility)}；说明：最近窗口激活波动。",
        f"- confidence_mean：{_fmt_num(summary.confidence_mean)}；说明：最近窗口平均可靠度。",
        f"- face_scale_mean：{_fmt_num(summary.face_scale_mean)}；face_scale_slope：{_fmt_signed(summary.face_scale_slope)}；说明：最近窗口中人脸距离变化趋势。",
        f"- blink_rate_mean：{_fmt_num(summary.blink_rate_mean, 2)} 次/分钟；当前相对滚动基线偏移：{_fmt_signed(summary.blink_rate_delta_current_vs_baseline, 2)}",
        f"- 当前窗口相对上一窗口差分：{summary.current_window_delta_vs_prev}",
        f"- 最近窗口主要 AU：{summary.top_aus or []}",
        f"- 最近窗口主要细粒度线索：{summary.subtle_top or []}",
    ]
    if summary.accepted_micro_expression_events:
        lines.append(f"- 最近窗口已接纳微表情事件：{len(summary.accepted_micro_expression_events)} 个")
    return "\n".join(lines)


def format_rmes_debug_cn(events: list[RMESSpotEvent], *, accepted: bool) -> str:
    if not events:
        return "无微表情事件。"
    status = "主线已接纳" if accepted else "实验事件（未接入主线）"
    lines = [status]
    for event in events[-6:]:
        lines.append(
            f"- window={event.window_id}; score={_fmt_num(event.spot_score)}; conf={_fmt_num(event.confidence)}; quality={event.quality_gate_passed}; emotion={event.dominant_emotion}; delta(valence={_fmt_signed(event.valence_delta)}, arousal={_fmt_signed(event.arousal_delta)})"
        )
        if event.interpretation:
            lines.append(f"  说明：{event.interpretation}")
    return "\n".join(lines)


def format_compact_context_cn(snapshot: StructuredContextSnapshot) -> str:
    parts = ["[本地结构化情绪上下文]", f"后端={snapshot.backend}({snapshot.backend_description})"]
    current = snapshot.current_window
    if current is not None:
        parts.extend(
            [
                f"当前窗口#{current.window_id}: valence_mean={_fmt_signed(current.valence_mean, 2)}, arousal_mean={_fmt_num(current.arousal_mean, 2)}, confidence_mean={_fmt_num(current.confidence_mean, 2)}, frontal_mean={_fmt_num(current.frontal_score_mean, 2)}",
                f"当前窗口scale: face_scale_mean={_fmt_num(current.face_scale_mean, 2)}, scale_bucket={current.dominant_scale_bucket}",
                f"当前窗口blink: count={current.blink_signal.blink_count}, rate={_fmt_num(current.blink_signal.blink_rate_per_min, 1)}/min, ibi={_fmt_num(current.blink_signal.mean_inter_blink_interval, 2)}s",
                f"当前窗口dominant_distribution={current.dominant_affect_distribution}",
                f"当前窗口top_aus={current.top_aus or []}",
                f"当前窗口subtle_top={current.subtle_top or []}",
            ]
        )
    if snapshot.rolling_summary is not None:
        rolling = snapshot.rolling_summary
        parts.extend(
            [
                f"最近{rolling.window_count}窗口: valence_mean={_fmt_signed(rolling.valence_mean, 2)}, valence_slope={_fmt_signed(rolling.valence_slope, 2)}, arousal_mean={_fmt_num(rolling.arousal_mean, 2)}, arousal_slope={_fmt_signed(rolling.arousal_slope, 2)}, confidence_mean={_fmt_num(rolling.confidence_mean, 2)}",
                f"最近{rolling.window_count}窗口scale_mean={_fmt_num(rolling.face_scale_mean, 2)}, scale_slope={_fmt_signed(rolling.face_scale_slope, 2)}",
                f"最近{rolling.window_count}窗口blink_rate_mean={_fmt_num(rolling.blink_rate_mean, 1)}/min, blink_delta_current_vs_baseline={_fmt_signed(rolling.blink_rate_delta_current_vs_baseline, 2)}",
            ]
        )
    if snapshot.accepted_micro_expression_events:
        parts.append(f"micro_expression_events={len(snapshot.accepted_micro_expression_events)}")
    parts.append("[/本地结构化情绪上下文]")
    return "\n".join(parts)
