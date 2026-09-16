"""Phase 2 figures and the markdown report (planned figures 1–4 and tables).

Every input comes from ``phase2.inference.analyze`` output and the per-cell
``val_curve.csv`` files; nothing is typed in.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_COLORS = {"main": "#1f77b4", "c1_span": "#ff7f0e", "c2_word": "#2ca02c"}
FAMILY_MARKERS = {"tesseract": "o", "unrolled": "s", "param_matched": "^", "width_scaled": "D"}
ROLE_FILL = {"primary_depth": "full", "diagnostic": "none", "anchor": "full", "unassigned": "none"}


def _role(analysis: Dict, t) -> str:
    roles = analysis["t_roles"]
    return roles.get(t, roles.get(str(t), "unassigned"))


def _mean_err(summary):
    if not summary:
        return None, None
    if summary.get("low") is None:
        return summary["mean"], None
    return summary["mean"], [[summary["mean"] - summary["low"]], [summary["high"] - summary["mean"]]]


def _placeholder(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="gray")


def plot_delta_vs_t(analysis: Dict, path: Path, checkpoint: str = "best") -> Path:
    block = analysis["checkpoints"][checkpoint]
    fig, ax = plt.subplots(figsize=(8, 5))
    primary = [int(t) for t, role in analysis["t_roles"].items() if role == "primary_depth"]
    if primary:
        ax.axvspan(math.log2(min(primary)) - 0.25, math.log2(max(primary)) + 0.25, color="#eef3fb", zorder=0,
                   label="primary depth T")
    plotted = False
    for task, per_t in sorted(block["delta"].items()):
        xs, ys = [], []
        for t, entry in sorted(per_t.items(), key=lambda kv: int(kv[0])):
            mean, err = _mean_err(entry["summary"])
            if mean is None:
                continue
            x = math.log2(int(t))
            xs.append(x)
            ys.append(mean)
            ax.errorbar([x], [mean], yerr=err, fmt="o", color=TASK_COLORS.get(task, "k"), capsize=4,
                        fillstyle=ROLE_FILL[_role(analysis, int(t))])
        if xs:
            ax.plot(xs, ys, color=TASK_COLORS.get(task, "k"), label=task)
            plotted = True
    if not plotted:
        _placeholder(ax, "no paired K-effect data")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["1 (anchor)", "2 (diagnostic)", "4", "8"])
    ax.set_xlabel("Transformation depth T")
    ax.set_ylabel(f"Δ(T) = score(K={analysis['decision_rule_config']['k_high']}) − "
                  f"score(K={analysis['decision_rule_config']['k_low']})")
    ax.set_title(f"K effect vs T (test chance-normalised accuracy, {checkpoint} checkpoint, 95 % CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_score_vs_k(analysis: Dict, path: Path, checkpoint: str = "best") -> Path:
    cells = [c for c in analysis["checkpoints"][checkpoint]["cells"] if c["family"] == "tesseract"]
    ts = sorted({c["t"] for c in cells})
    fig, axes = plt.subplots(1, max(len(ts), 1), figsize=(4 * max(len(ts), 1), 4), squeeze=False)
    for ax, t in zip(axes[0], ts or [None]):
        if t is None:
            _placeholder(ax, "no Tesseract cells")
            continue
        for task in sorted({c["task"] for c in cells if c["t"] == t}):
            rows = sorted((c for c in cells if c["t"] == t and c["task"] == task and c["score"]), key=lambda c: c["depth"])
            for c in rows:
                mean, err = _mean_err(c["score"])
                ax.errorbar([math.log2(c["depth"])], [mean], yerr=err, fmt="o", color=TASK_COLORS.get(task, "k"), capsize=3)
            ax.plot([math.log2(c["depth"]) for c in rows], [c["score"]["mean"] for c in rows],
                    color=TASK_COLORS.get(task, "k"), label=task)
        ax.set_title(f"T={t} ({_role(analysis, t)})")
        ax.set_xlabel("log2 K")
        ax.set_ylabel("test chance-normalised accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_score_vs_flops(analysis: Dict, path: Path, checkpoint: str = "best") -> Path:
    per_t = analysis["checkpoints"][checkpoint]["compute_normalised"]
    ts = sorted(per_t, key=int)
    fig, axes = plt.subplots(1, max(len(ts), 1), figsize=(5 * max(len(ts), 1), 4), squeeze=False)
    for ax, t in zip(axes[0], ts or [None]):
        if t is None:
            _placeholder(ax, "no main-task cells")
            continue
        for p in per_t[t]["points"]:
            if not p["score"]:
                continue
            mean, err = _mean_err(p["score"])
            ax.errorbar([p["forward_flops_per_sequence"]], [mean], yerr=err, fmt=FAMILY_MARKERS.get(p["family"], "x"),
                        capsize=3, label=f"{p['family']} d={p['depth']}")
        ax.set_xscale("log")
        ax.set_title(f"T={t}: score vs forward FLOPs/sequence")
        ax.set_xlabel("forward FLOPs per sequence")
        ax.set_ylabel("test chance-normalised accuracy")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def load_curves(cells_root: Path) -> Dict[str, List[List[Dict]]]:
    """``"family/task/T/depth"`` → one validation curve per seed."""
    curves: Dict[str, List[List[Dict]]] = defaultdict(list)
    for result_path in sorted(Path(cells_root).glob("*/result.json")):
        cell = json.loads(result_path.read_text(encoding="utf-8"))["cell"]
        with open(result_path.parent / "val_curve.csv", newline="", encoding="utf-8") as f:
            rows = [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(f)]
        curves[f"{cell['family']}/{cell['task']}/{cell['t']}/{cell['depth']}"].append(rows)
    return dict(curves)


def plot_learning_curves(curves: Dict[str, List[List[Dict]]], path: Path) -> Path:
    keys = [k for k in curves if k.startswith("tesseract/main/")]
    ts = sorted({int(k.split("/")[2]) for k in keys})
    fig, axes = plt.subplots(1, max(len(ts), 1), figsize=(4.5 * max(len(ts), 1), 4), squeeze=False)
    for ax, t in zip(axes[0], ts or [None]):
        if t is None:
            _placeholder(ax, "no learning curves")
            continue
        for key in sorted((k for k in keys if int(k.split("/")[2]) == t), key=lambda k: int(k.split("/")[3])):
            seeds = [c for c in curves[key] if c]
            length = min(len(c) for c in seeds)
            steps = [seeds[0][i]["step"] for i in range(length)]
            values = [[c[i]["val_chance_normalised_token_accuracy"] for c in seeds] for i in range(length)]
            mean = [sum(v) / len(v) for v in values]
            line, = ax.plot(steps, mean, label=f"K={key.split('/')[3]}")
            ax.fill_between(steps, [min(v) for v in values], [max(v) for v in values], color=line.get_color(), alpha=0.2)
        ax.set_title(f"T={t}: validation curves (seed mean, min–max band)")
        ax.set_xlabel("step")
        ax.set_ylabel("val chance-normalised accuracy")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmt(summary) -> str:
    if not summary:
        return "—"
    if summary.get("low") is None:
        return f"{summary['mean']:.4f} (n=1)"
    return f"{summary['mean']:.4f} [{summary['low']:.4f}, {summary['high']:.4f}] (n={summary['n']})"


def report_markdown(analysis: Dict) -> str:
    lines = ["# Phase 2 analysis report", "",
             f"Results analysed: {analysis['n_results']}. Primary checkpoint: best-validation; sensitivity: final.",
             "", "Known confounds: " + "; ".join(analysis["known_confounds"]), ""]
    for checkpoint in ("best", "final"):
        block = analysis["checkpoints"][checkpoint]
        rule = block["decision_rule"]
        lines += [f"## {checkpoint} checkpoint", "", f"**Decision rule outcome (Amendment 02 D1): `{rule['outcome']}`**", ""]
        lines += [f"- {name}: {value}" for name, value in rule["conditions"].items()]
        lines += [f"- flags: {rule['flags']}", "", "| Task | T | role | Δ(T) mean [95% CI] | K* |", "|---|---|---|---|---|"]
        for task, per_t in block["delta"].items():
            for t, entry in sorted(per_t.items(), key=lambda kv: int(kv[0])):
                lines.append(f"| {task} | {t} | {entry['role']} | {_fmt(entry['summary'])} | {block['k_star'][task].get(t)} |")
        lines += ["", "| Task | DiD (primary T) |", "|---|---|"]
        lines += [f"| {task} | {_fmt(d['summary'])} |" for task, d in block["did_primary"].items()]
        lines += ["", "Interaction regressions (primary T only): " + json.dumps(block["regression_primary_t"], default=str), ""]
        lines += ["| Baseline comparison | T | mean [95% CI] | notes |", "|---|---|---|---|"]
        for t, entries in block["baselines"].items():
            for name, comparison in entries.items():
                note = (f"head_dim {comparison['head_dim']} vs {comparison['tesseract_head_dim']}"
                        if "head_dim" in comparison else "")
                lines.append(f"| {name} | {t} | {_fmt(comparison['summary'])} | {note} |")
        lines += ["", "| Cell | role | seeds | score | EM | params | FLOPs/seq | head_dim |", "|---|---|---|---|---|---|---|---|"]
        for c in block["cells"]:
            lines.append(f"| {c['family']}/{c['task']}/T{c['t']}/d{c['depth']} | {c['role']} | {c['seeds']} | {_fmt(c['score'])} | "
                         f"{_fmt(c['exact_match'])} | {c['parameter_count']:,} | {c['forward_flops_per_sequence']:,} | {c['head_dim']} |")
        floors = [k for k, v in block["floor_ceiling"].items() if v["floor"]]
        lines += ["", f"Cells at floor (all K and seeds): {floors or 'none'}", ""]
    return "\n".join(lines)


def write_report(analysis: Dict, out_dir: Path, curves: Dict[str, List[List[Dict]]]) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_delta_vs_t(analysis, out_dir / "fig1_delta_vs_t.png"),
        plot_score_vs_k(analysis, out_dir / "fig2_score_vs_k.png"),
        plot_score_vs_flops(analysis, out_dir / "fig3_score_vs_flops.png"),
        plot_learning_curves(curves, out_dir / "fig4_learning_curves.png"),
    ]
    report = out_dir / "report.md"
    report.write_text(report_markdown(analysis) + "\n", encoding="utf-8")
    return paths + [report]
