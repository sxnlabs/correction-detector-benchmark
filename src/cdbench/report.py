"""Aggregate scores into results/: summary, table, plots. Aggregates only."""

import json
from pathlib import Path

from cdbench.dataset import Message
from cdbench.metrics import bootstrap_ci, paired_bootstrap_diff, scores
from cdbench.privacy import assert_no_message_text

DETECTORS = ("regex", "haiku", "haiku+prior", "haiku vote3", "haiku+prior vote3", "pipeline",
             "tfidf", "tfidf+prior", "finetune", "finetune+prior")
# The pairs the conclusion rests on, compared on the same messages.
COMPARISONS = (("haiku", "regex"), ("pipeline", "haiku+prior"), ("haiku", "tfidf"),
               ("haiku", "finetune"), ("tfidf", "finetune"),
               ("haiku+prior vote3", "haiku"), ("tfidf", "tfidf+prior"),
               ("finetune", "finetune+prior"))


def summarize(gold: dict[str, bool], preds: dict[str, dict[str, bool]], extra: dict) -> dict:
    ids = sorted(gold)
    y_true = [gold[i] for i in ids]
    detectors = {}
    for name in DETECTORS:
        if name not in preds:
            continue
        y_pred = [bool(preds[name].get(i, False)) for i in ids]
        row = scores(y_true, y_pred)
        for metric in ("precision", "recall", "f1"):
            row[f"{metric}_ci"] = list(bootstrap_ci(y_true, y_pred, metric))
        detectors[name] = row
    differences = {}
    for a, b in COMPARISONS:
        if a in preds and b in preds:
            diff, low, high = paired_bootstrap_diff(
                y_true, [bool(preds[a].get(i, False)) for i in ids],
                [bool(preds[b].get(i, False)) for i in ids], "f1")
            differences[f"{a} - {b}"] = {"diff": diff, "ci": [low, high]}
    return {"detectors": detectors, "f1_differences": differences, **extra}


def _fmt(row: dict, metric: str) -> str:
    low, high = row[f"{metric}_ci"]
    return f"{row[metric]:.2f} [{low:.2f}, {high:.2f}]"


def render_table(summary: dict) -> str:
    cli = summary.get("cost_cli_per_1000", {})
    direct = summary.get("cost_direct_per_1000", {})
    latency = summary.get("latency", {})
    header = ("| Detector | Precision [95 % CI] | Recall [95 % CI] | F1 [95 % CI] | "
              "Cost / 1,000 msgs, direct API (claude -p) | Latency p50 / p95 |")
    lines = [header,
             "|---|---|---|---|---|---|"]
    for name, row in summary["detectors"].items():
        cost = "$0"
        if name in direct:
            cost = f"${direct[name]:.4f} (${cli.get(name, 0):.3f})"
        lat = latency.get(name)
        lat_s = f"{lat['p50'] * 1000:.1f} ms / {lat['p95'] * 1000:.1f} ms" if lat else "n/a"
        lines.append(f"| {name} | {_fmt(row, 'precision')} | {_fmt(row, 'recall')} | "
                     f"{_fmt(row, 'f1')} | {cost} | {lat_s} |")
    diffs = summary.get("f1_differences", {})
    if diffs:
        lines += ["", "| Pair | F1 difference [95 % CI, paired] |", "|---|---|"]
        for pair, d in diffs.items():
            low, high = d["ci"]
            lines.append(f"| {pair} | {d['diff']:+.2f} [{low:+.2f}, {high:+.2f}] |")
    return "\n".join(lines) + "\n"


def write_report(summary: dict, meta: dict, out_dir: Path, *, messages: list[Message],
                 extra_paths: list[Path], probas: dict | None = None,
                 gold: dict[str, bool] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "table.md").write_text(render_table(summary))
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    if probas and gold:
        _plots(summary, probas, gold, out_dir)
    assert_no_message_text([out_dir, *extra_paths], messages)


def _plots(summary: dict, probas: dict, gold: dict[str, bool], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    names = list(summary["detectors"])
    fig, axes = plt.subplots(2, (len(names) + 1) // 2, figsize=(3 * ((len(names) + 1) // 2), 6))
    for ax, name in zip(axes.flat, names, strict=False):
        r = summary["detectors"][name]
        ax.imshow([[r["tp"], r["fn"]], [r["fp"], r["tn"]]], cmap="Blues")
        for (y, x), v in {(0, 0): r["tp"], (0, 1): r["fn"], (1, 0): r["fp"],
                          (1, 1): r["tn"]}.items():
            ax.text(x, y, str(v), ha="center", va="center")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([0, 1], ["pred +", "pred -"])
        ax.set_yticks([0, 1], ["true +", "true -"])
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ids = sorted(gold)
    for name, by_id in probas.items():
        precision, recall, _ = precision_recall_curve([gold[i] for i in ids],
                                                      [by_id[i] for i in ids])
        ax.plot(recall, precision, label=name)
    for name in ("regex", "haiku+prior", "pipeline"):
        if name in summary["detectors"]:
            r = summary["detectors"][name]
            ax.scatter([r["recall"]], [r["precision"]], label=name, marker="x")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=120)
    plt.close(fig)
