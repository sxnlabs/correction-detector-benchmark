"""`cdbench` command line: each subcommand runs one stage on the private data dir."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cdbench.dataset import load_messages, snapshot
from cdbench.detectors.haiku import haiku_label, haiku_predict
from cdbench.detectors.regex import regex_predict
from cdbench.haiku_runs import measure_overhead, run_haiku
from cdbench.labeling.arbitration import (
    ReviewItem,
    disagreement_counts,
    merge_gold,
    select_for_review,
)
from cdbench.labeling.opus import OpusLabel, label_with_opus
from cdbench.store import read_rows, write_rows


def _cmd_snapshot(args: argparse.Namespace) -> None:
    snapshot(Path(args.chat_log).expanduser(), Path(args.data_dir).expanduser())
    msgs = load_messages(Path(args.data_dir).expanduser())
    eligible = sum(m.llm_eligible for m in msgs)
    print(f"in scope: {len(msgs)}, llm eligible: {eligible}")


def _data(args: argparse.Namespace) -> Path:
    return Path(args.data_dir).expanduser()


def _cmd_label(args: argparse.Namespace) -> None:
    data = _data(args)
    labels, failures = label_with_opus(load_messages(data), cache_dir=data / "cache",
                                       workers=args.workers)
    write_rows(data / "opus.jsonl", labels)
    print(f"labeled: {len(labels)}, corrections: {sum(x.label for x in labels)}, "
          f"low confidence: {sum(x.confidence < 0.7 for x in labels)}, failed: {len(failures)}")


def _cmd_select_review(args: argparse.Namespace) -> None:
    data = _data(args)
    messages = load_messages(data)
    opus = read_rows(data / "opus.jsonl", OpusLabel)
    regex = {m.id: regex_predict(m.text) for m in messages}

    def first_run(m):
        if not m.llm_eligible:
            return m.id, False
        call = haiku_predict(m, variant="with_prior", run=0, cache_dir=data / "cache")
        return m.id, haiku_label(call)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        haiku = dict(pool.map(first_run, messages))
    items = select_for_review(opus, regex, haiku)
    write_rows(data / "review.jsonl", items)
    (data / "review_meta.json").write_text(json.dumps(
        {"disagreements_before_cap": disagreement_counts(opus, regex, haiku)}))
    counts = {}
    for it in items:
        counts[it.reason] = counts.get(it.reason, 0) + 1
    print(f"to review: {len(items)} {counts}, haiku failures: "
          f"{sum(v is None for v in haiku.values())}")


def _cmd_merge_gold(args: argparse.Namespace) -> None:
    data = _data(args)
    opus = read_rows(data / "opus.jsonl", OpusLabel)
    reviewed = {it.msg_id for it in read_rows(data / "review.jsonl", ReviewItem)}
    answers_path = data / "answers.json"
    answers = json.loads(answers_path.read_text()) if answers_path.exists() else {}
    gold = merge_gold(opus, answers, reviewed)
    write_rows(data / "gold.jsonl", gold)
    by = {}
    for g in gold:
        by[g.resolved_by] = by.get(g.resolved_by, 0) + 1
    print(f"gold: {len(gold)}, corrections: {sum(g.label for g in gold)}, resolved_by: {by}")


def _cmd_run_haiku(args: argparse.Namespace) -> None:
    data = _data(args)
    variants = tuple(v.strip() for v in args.variants.split(","))
    rows = run_haiku(load_messages(data), runs=args.runs, variants=variants,
                     cache_dir=data / "cache", workers=args.workers)
    write_rows(data / "haiku_runs.jsonl", rows)
    overhead = measure_overhead(data / "cache")
    (data / "haiku_overhead.json").write_text(f'{{"overhead_input_tokens": {overhead}}}')
    models = sorted({r.model for r in rows if r.error is None})
    print(f"runs: {len(rows)}, errors: {sum(r.error is not None for r in rows)}, "
          f"models: {models}, cli overhead tokens: {overhead}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from cdbench.evaluate import evaluate
    from cdbench.report import write_report

    data = _data(args)
    repo = Path(__file__).resolve().parents[2]
    summary, meta, probas, gold = evaluate(
        data, finetune=not args.no_finetune,
        finetune_model=str(Path(args.finetune_model).expanduser()) if args.finetune_model else None,
        epochs=args.epochs)
    write_report(summary, meta, repo / "results", messages=load_messages(data),
                 extra_paths=[repo / "README.md"], probas=probas, gold=gold)
    print((repo / "results" / "table.md").read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cdbench")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("snapshot", help="freeze the chat log into the data dir")
    p.add_argument("--chat-log", required=True)
    p.add_argument("--data-dir", required=True)
    p.set_defaults(func=_cmd_snapshot)
    for name, func, help_text in (
        ("label", _cmd_label, "first-pass labels by Opus"),
        ("select-review", _cmd_select_review, "pick the messages Nathan arbitrates"),
        ("merge-gold", _cmd_merge_gold, "merge Opus labels and Nathan's answers"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--data-dir", required=True)
        p.add_argument("--workers", type=int, default=4)
        p.set_defaults(func=func)
    p = sub.add_parser("run-haiku", help="repeated Haiku runs on eligible messages")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--variants", default="message,with_prior")
    p.set_defaults(func=_cmd_run_haiku)
    p = sub.add_parser("evaluate", help="score every detector and write results/")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--finetune-model", help="local model directory (default: hub id)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--no-finetune", action="store_true")
    p.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
