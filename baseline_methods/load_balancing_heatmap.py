#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate CA selection heatmaps from experiment result JSON."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import patheffects
from matplotlib.ticker import FuncFormatter
import matplotlib.font_manager as fm


RESULT_DIR = Path(__file__).parent / "results" / "load_balancing"
DEFAULT_SUMMARY_PATH = RESULT_DIR / "selection_summary.json"
DEFAULT_HEATMAP_PATH = RESULT_DIR / "selection_heatmap.pdf"
DEFAULT_DRL_COMPARISON_PATH = RESULT_DIR / "drl_model_comparison.json"
DEFAULT_DRL_HEATMAP_PATH = RESULT_DIR / "drl_model_comparison_heatmap.pdf"
DEFAULT_CA_SCALE_SENSITIVITY_PATH = RESULT_DIR / "ca_scale_sensitivity.json"
DEFAULT_CA_SCALE_CURVE_PATH = RESULT_DIR / "ca_scale_sensitivity_rank_load_lorenz.pdf"


# Prefer loading project-local custom fonts first
font_dir = Path(__file__).resolve().parent.parent / 'font'
if font_dir.is_dir():
    font_files = sorted(font_dir.glob('*.ttf'))
    for font_path in font_files:
        try:
            fm.fontManager.addfont(str(font_path))
        except Exception as exc:
            print(f"Font load failed {font_path.name}: {exc}")


NAME_MAP: Dict[str, str] = {
    "random selection": "Random",
    "random": "Random",
    "random selection strategy": "Random",
    "weighted score": "Weighted",
    "weighted": "Weighted",
    "weighted score strategy": "Weighted",
    "round robin": "Round_Robin",
    "round robin fixed random order": "Round_Robin",
    "round robin strategy": "Round_Robin",
    "reputation": "Reputation",
    "reputation based": "Reputation",
    "reputation based strategy": "Reputation",
    "epsilon greedy": "Epsilon_Greedy",
    "epsilon greedy strategy": "Epsilon_Greedy",
    "partial semi-greedy": "PSG",
    "partial semi greedy": "PSG",
    "psg": "PSG",
    "partial semi greedy strategy": "PSG",
    "genetic algorithm": "GA",
    "ga": "GA",
    "genetic algorithm strategy": "GA",
    "ddqn model selection": "DRL-DPKI",
    "ddqn": "DRL-DPKI",
    "ddqn model": "DRL-DPKI",
    "drl-dpki": "DRL-DPKI",
    "drl dpki": "DRL-DPKI",
}


def normalize_strategy_name(raw: str) -> str:
    base = raw.lower().replace("-", " ").replace("_", " ")
    base = " ".join(base.split())
    return NAME_MAP.get(base, raw)


def load_summary(
    path: Path,
    use_frequencies: bool = False,
) -> Tuple[List[str], List[List[float]], List[int], int]:
    """Load experiment results and return the data required for plotting."""
    with path.open("r", encoding="utf-8") as in_file:
        summary = json.load(in_file)

    results = summary.get("results", {})
    if not results:
        raise ValueError("No valid data found in the result file")

    n_na = summary.get("n_na", 0)

    labels: List[str] = []
    counts: List[List[float]] = []
    value_key = "selection_frequencies" if use_frequencies else "selection_counts"

    for label, payload in results.items():
        selection_values = payload.get(value_key)
        if selection_values is None:
            if use_frequencies and payload.get("selection_counts") is not None:
                print(f"Strategy {label}: missing selection_frequencies; falling back to selection_counts")
                selection_values = payload.get("selection_counts")
                value_key = "selection_counts"
            else:
                print(f"Skipping strategy {label}: missing {value_key}")
                continue

        if n_na and len(selection_values) != n_na:
            print(
                f"Skipping strategy {label}: {value_key} length {len(selection_values)} does not match n_na {n_na}"
            )
            continue

        if not n_na:
            n_na = len(selection_values)

        strategy_name = payload.get("strategy_name", label)
        if "(" in strategy_name:
            strategy_name = strategy_name.split("(", 1)[0].strip()
        normalized_name = normalize_strategy_name(strategy_name)
        if normalized_name == "Epsilon_Greedy":
            continue
        labels.append(normalized_name)
        counts.append([float(x) for x in selection_values])

    if not counts:
        raise ValueError("No valid strategy data found for plotting")

    if "Round_Robin" in labels and "Reputation" in labels:
        rr_idx = labels.index("Round_Robin")
        rep_idx = labels.index("Reputation")
        if rep_idx != rr_idx + 1:
            rep_label = labels.pop(rep_idx)
            rep_counts = counts.pop(rep_idx)
            if rep_idx < rr_idx:
                rr_idx -= 1
            labels.insert(rr_idx + 1, rep_label)
            counts.insert(rr_idx + 1, rep_counts)

    malicious_nas = summary.get("malicious_nas", [])
    return labels, counts, malicious_nas, n_na


def load_drl_comparison_summary(
    path: Path,
    use_frequencies: bool = False,
) -> Tuple[List[str], List[List[float]], List[int], int]:
    with path.open("r", encoding="utf-8") as in_file:
        summary = json.load(in_file)

    results = summary.get("results", {})
    if not results:
        raise ValueError("No valid data found in the result file")

    baseline_key = "drl_baseline_balance"
    ablation_key = "drl_ablation_no_hunger"

    baseline = results.get(baseline_key)
    ablation = results.get(ablation_key)
    if baseline is None or ablation is None:
        raise ValueError(f"Comparison results missing strategies: {baseline_key} / {ablation_key}")

    value_key = "selection_frequencies" if use_frequencies else "selection_counts"

    baseline_counts = baseline.get(value_key)
    ablation_counts = ablation.get(value_key)

    if baseline_counts is None or ablation_counts is None:
        if use_frequencies and baseline.get("selection_counts") is not None and ablation.get("selection_counts") is not None:
            print("Comparison results missing selection_frequencies; falling back to selection_counts")
            baseline_counts = baseline.get("selection_counts")
            ablation_counts = ablation.get("selection_counts")
            value_key = "selection_counts"
        else:
            raise ValueError(f"Comparison results missing {value_key}")

    n_na = summary.get("n_na", 0) or len(baseline_counts)
    if len(baseline_counts) != n_na or len(ablation_counts) != n_na:
        raise ValueError("selection_counts length does not match n_na")

    malicious_nas = summary.get("malicious_nas", [])
    labels = ["DRL-DPKI (w/o hunger)", "DRL-DPKI (Full)"]
    counts = [[float(x) for x in ablation_counts], [float(x) for x in baseline_counts]]
    return labels, counts, malicious_nas, n_na


def _load_share(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=float)
    return values.astype(float) / total


def jain_fairness(values: np.ndarray) -> float:
    x = values.astype(float)
    denom = float(x.size) * float(np.square(x).sum())
    if denom <= 0:
        return 1.0
    return float(np.square(x.sum()) / denom)


def gini_coefficient(values: np.ndarray) -> float:
    x = values.astype(float)
    total = float(x.sum())
    if total <= 0 or x.size == 0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (index * x_sorted).sum()) / (n * total) - (n + 1.0) / n)


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size < 3:
        return x
    w = int(window)
    if w < 3:
        return x
    if w % 2 == 0:
        w += 1
    w = min(w, int(x.size) if int(x.size) % 2 == 1 else int(x.size) - 1)
    if w < 3:
        return x

    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(xp, kernel, mode="valid")


def load_ca_scale_sensitivity_summary(
    path: Path,
    use_frequencies: bool = False,
) -> Tuple[List[int], List[str], Dict[str, Dict[int, np.ndarray]]]:
    with path.open("r", encoding="utf-8") as in_file:
        summary = json.load(in_file)

    results_by_ca_count = summary.get("results_by_ca_count", {})
    if not results_by_ca_count:
        raise ValueError("No valid results_by_ca_count found in the result file")

    ca_counts_raw = summary.get("ca_counts")
    if ca_counts_raw:
        ca_counts = [int(x) for x in ca_counts_raw]
    else:
        ca_counts = sorted(int(k) for k in results_by_ca_count.keys())

    value_key = "selection_frequencies" if use_frequencies else "selection_counts"

    strategy_order: List[str] = []
    data: Dict[str, Dict[int, np.ndarray]] = {}

    for idx, n_na in enumerate(ca_counts):
        per_summary = results_by_ca_count.get(str(n_na))
        if per_summary is None:
            continue
        results = per_summary.get("results", {})
        if not results:
            continue

        for label, payload in results.items():
            selection_values = payload.get(value_key)
            if selection_values is None:
                if use_frequencies and payload.get("selection_counts") is not None:
                    selection_values = payload.get("selection_counts")
                else:
                    continue

            strategy_name = payload.get("strategy_name", label)
            if "(" in strategy_name:
                strategy_name = strategy_name.split("(", 1)[0].strip()
            normalized_name = normalize_strategy_name(strategy_name)
            if normalized_name == "Epsilon_Greedy":
                continue

            arr = np.asarray(selection_values, dtype=float)
            data.setdefault(normalized_name, {})[n_na] = arr

            if idx == 0 and normalized_name not in strategy_order:
                strategy_order.append(normalized_name)

    if "Round_Robin" in strategy_order and "Reputation" in strategy_order:
        rr_idx = strategy_order.index("Round_Robin")
        rep_idx = strategy_order.index("Reputation")
        if rep_idx != rr_idx + 1:
            rep_label = strategy_order.pop(rep_idx)
            if rep_idx < rr_idx:
                rr_idx -= 1
            strategy_order.insert(rr_idx + 1, rep_label)

    if not strategy_order:
        raise ValueError("No valid strategy data found for plotting")

    return ca_counts, strategy_order, data


def plot_rank_load_lorenz(
    ca_counts: List[int],
    strategy_order: List[str],
    data: Dict[str, Dict[int, np.ndarray]],
    figure_path: Path,
    title: str = "",
) -> None:
    label_fontsize = 34
    tick_fontsize = 17
    title_fontsize = 30
    legend_fontsize = 24

    plt.rcParams.update({
        "font.family": ["Times New Roman", "serif"],
        "font.serif": ["Times New Roman", "Times", "serif"],
        "font.size": label_fontsize,
        "axes.titlesize": title_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "legend.fontsize": legend_fontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    n_rows = len(strategy_order)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(12, max(2, 4.5 * n_rows)),
        sharex=True,
    )

    axes = np.atleast_1d(axes)

    default_colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    if not default_colors:
        default_colors = [plt.cm.get_cmap("tab10")(i) for i in range(10)]

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]

    legend_lines: List[Line2D] = []

    for row_idx, strategy in enumerate(strategy_order):
        ax_rank = axes[row_idx]

        for m_idx, n_na in enumerate(ca_counts):
            values = data.get(strategy, {}).get(n_na)
            if values is None:
                continue

            color = default_colors[m_idx % len(default_colors)]
            marker = markers[m_idx % len(markers)]

            share = _load_share(values)
            x_idx = np.array([0.0]) if share.size <= 1 else np.linspace(0.0, 100.0, share.size)
            markevery = max(1, int(round(share.size / 40)))
            (line_rank,) = ax_rank.plot(
                x_idx,
                share,
                linewidth=2,
                linestyle="-",
                marker=marker,
                markersize=6,
                markevery=markevery,
                color=color,
            )

            if row_idx == 0:
                legend_lines.append(line_rank)

        ax_rank.set_ylabel("Selection frequency (%)", fontsize=17)
        ax_rank.set_xlim(0.0, 100.0)
        ax_rank.set_ylim(bottom=0.0)
        ax_rank.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f}%"))
        ax_rank.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("CA Index Percentile (%)", fontsize=17)

    labels = [f"M={n}" for n in ca_counts]
    if legend_lines:
        axes[0].legend(
            legend_lines,
            labels,
            loc="upper right",
            ncol=1,
            fontsize=16,
            frameon=True,
            framealpha=0.85,
            borderpad=0.4,
            handlelength=1.8,
            handletextpad=0.6,
            labelspacing=0.3,
        )

    if title:
        fig.suptitle(title, fontsize=18, y=0.995)
        fig.tight_layout(pad=0.2)
    else:
        fig.tight_layout(pad=0.2)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_heatmap(
    labels: List[str],
    counts: List[List[float]],
    malicious_nas: List[int],
    n_na: int,
    figure_path: Path,
    title: str,
    is_frequency: bool = False,
) -> None:
    """Plot and save the heatmap."""

    # Use Times New Roman and embed fonts for vector PDF output
    plt.rcParams.update({
        "font.family": ["Times New Roman", "serif"],
        "font.serif": ["Times New Roman", "Times", "serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    heatmap_data = np.array(counts)
    width_per_ca = 1.15 if is_frequency else 0.8
    fig_width = max(8, n_na * width_per_ca)
    fig_height = 4 + 0.5 * len(labels)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="Blues")

    if title:
        ax.set_title(title, fontsize=32)
    ax.set_xlabel("CA Index", fontsize=28)
    ax.set_ylabel("Strategy", fontsize=28)
    ax.set_xticks(np.arange(n_na))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(np.arange(n_na), fontsize=28)
    ax.set_yticklabels(labels, fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=28)

    if malicious_nas:
        for idx in malicious_nas:
            if 0 <= idx < n_na:
                ax.axvline(x=idx, color="red", linestyle="--", linewidth=1.0)
        legend_proxy = Line2D([0], [0], color="red", linestyle="--", linewidth=2.0, label="Malicious CA")
        ax.legend(handles=[legend_proxy], loc="upper left", fontsize=20)

    max_value = heatmap_data.max() if heatmap_data.size > 0 else 0
    threshold = max_value * 0.5
    for row_idx in range(heatmap_data.shape[0]):
        for col_idx in range(heatmap_data.shape[1]):
            raw_value = float(heatmap_data[row_idx, col_idx])
            color = "white" if raw_value > threshold else "black"
            if is_frequency:
                display_value = f"{raw_value * 100:.1f}%"
            else:
                display_value = str(int(raw_value))
            cell_fontsize = 22 if is_frequency else 22
            if n_na >= 25:
                cell_fontsize = max(12, cell_fontsize - 4)
            text = ax.text(
                col_idx,
                row_idx,
                display_value,
                ha="center",
                va="center",
                color=color,
                fontsize=cell_fontsize,
            )
            outline_color = "black" if color == "white" else "white"
            text.set_path_effects([patheffects.withStroke(linewidth=2, foreground=outline_color)])

    cbar_label = "Selection Frequency (%)" if is_frequency else "Selection Count"
    cbar = fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.04, pad=0.015)
    if is_frequency:
        cbar.formatter = FuncFormatter(lambda x, pos: f"{x * 100:.0f}%")
        cbar.update_ticks()
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label(cbar_label, fontsize=26)
    fig.tight_layout(rect=[0, 0, 0.98, 1])

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load experiment results and generate a heatmap")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to the experiment result JSON file",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_HEATMAP_PATH,
        help="Output heatmap path (PDF)",
    )
    return parser.parse_args()


def _prompt_choice(prompt: str, default: str = "1") -> str:
    try:
        value = input(prompt).strip()
    except EOFError:
        value = ""
    return value or default


def choose_plot_mode() -> str:
    print("\n=== Load Balancing Heatmap ===")
    print("1) Baseline heatmap (existing logic)")
    print("2) DRL-DPKI ablation comparison (baseline vs no_hunger)")
    print("3) CA scale sensitivity: Index–Load curve + (Jain/Gini) metrics")
    return _prompt_choice("Select mode [1/2/3] (default=1): ", default="1")


def choose_value_mode() -> str:
    print("\n=== Heatmap Value ===")
    print("1) selection_counts")
    print("2) selection_frequencies")
    return _prompt_choice("Select value [1/2] (default=1): ", default="1")


def main() -> None:
    args = parse_args()

    mode = choose_plot_mode()
    value_mode = choose_value_mode()
    use_frequencies = (value_mode == "2")

    input_path = args.input
    figure_path = args.figure

    if mode == "3":
        if input_path == DEFAULT_SUMMARY_PATH:
            input_path = DEFAULT_CA_SCALE_SENSITIVITY_PATH
        if figure_path == DEFAULT_HEATMAP_PATH:
            figure_path = DEFAULT_CA_SCALE_CURVE_PATH

        ca_counts, strategy_order, data = load_ca_scale_sensitivity_summary(
            input_path,
            use_frequencies=use_frequencies,
        )
        plot_rank_load_lorenz(
            ca_counts=ca_counts,
            strategy_order=strategy_order,
            data=data,
            figure_path=figure_path,
            title="",
        )

        print("\n=== Fairness Metrics (per strategy, per M) ===")
        for strategy in strategy_order:
            parts = []
            for n_na in ca_counts:
                values = data.get(strategy, {}).get(n_na)
                if values is None:
                    continue
                share = _load_share(np.asarray(values, dtype=float))
                parts.append(
                    f"M={n_na}: Jain={jain_fairness(share):.4f}, Gini={gini_coefficient(share):.4f}"
                )
            if parts:
                print(f"{strategy}: " + " | ".join(parts))

        print(f"Curve figure saved to: {figure_path}")
        return

    if mode == "2":
        if input_path == DEFAULT_SUMMARY_PATH:
            input_path = DEFAULT_DRL_COMPARISON_PATH
        if figure_path == DEFAULT_HEATMAP_PATH:
            figure_path = DEFAULT_DRL_HEATMAP_PATH
        labels, counts, malicious_nas, n_na = load_drl_comparison_summary(input_path, use_frequencies=use_frequencies)
    else:
        labels, counts, malicious_nas, n_na = load_summary(input_path, use_frequencies=use_frequencies)

    plot_heatmap(labels, counts, malicious_nas, n_na, figure_path, title="", is_frequency=use_frequencies)
    print(f"Heatmap saved to: {figure_path}")


if __name__ == "__main__":
    main()
