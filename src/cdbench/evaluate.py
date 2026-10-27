"""Collect every detector's predictions on the gold set and the aggregates around them."""

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np

from cdbench.cost import (
    HAIKU_PRICE_INPUT_PER_MTOK,
    HAIKU_PRICE_OUTPUT_PER_MTOK,
    PRICE_READ_ON,
    PRICE_SOURCE,
    cost_per_1000,
    direct_input_tokens,
)
from cdbench.dataset import load_messages
from cdbench.detectors import tfidf
from cdbench.detectors.regex import regex_predict
from cdbench.folds import make_folds
from cdbench.haiku_runs import HaikuRun, majority
from cdbench.labeling.arbitration import SEED, GoldLabel, ReviewItem
from cdbench.labeling.opus import OpusLabel
from cdbench.metrics import latency_summary, scores, wilson_interval
from cdbench.report import summarize
from cdbench.store import read_rows

VARIANTS = {"message": "", "with_prior": "+prior"}


def _folds(data: Path, gold: list[GoldLabel]) -> list[list[str]]:
    """Folds fixed for a given set of gold labels; rebuilt when the labels change."""
    path = data / "folds.json"
    digest = hashlib.sha256(json.dumps(sorted((g.msg_id, g.label) for g in gold)).encode()
                            ).hexdigest()
    if path.exists():
        stored = json.loads(path.read_text())
        if isinstance(stored, dict) and stored.get("gold_hash") == digest:
            return stored["folds"]
    folds = make_folds(gold)
    path.write_text(json.dumps({"gold_hash": digest, "folds": folds}))
    return folds


def _haiku(runs: list[HaikuRun], ids: list[str], eligible: set[str], gold: dict[str, bool],
           overhead: int):
    """Headline rows use run 0 (one call per message, as in production); the vote of the
    three runs is a separate row priced three calls. Costs are per 1,000 in-scope messages:
    only eligible messages are sent to Haiku."""
    preds, spread, cli_cost, direct_cost, latency, per_call = {}, {}, {}, {}, {}, {}
    share = len(eligible) / len(ids)
    for variant, suffix in VARIANTS.items():
        name = f"haiku{suffix}"
        mine = [r for r in runs if r.variant == variant]
        if not mine:
            continue
        by_msg: dict[str, list] = {}
        for r in mine:
            by_msg.setdefault(r.msg_id, []).append(r.label)
        first = {r.msg_id: r.label for r in mine if r.run == 0}
        preds[name] = {i: bool(i in eligible and first.get(i)) for i in ids}
        preds[f"{name} vote3"] = {i: bool(i in eligible and majority(by_msg.get(i, [])))
                                  for i in ids}
        per_run = []
        for run in sorted({r.run for r in mine}):
            labels = {r.msg_id: r.label for r in mine if r.run == run}
            y_pred = [bool(i in eligible and labels.get(i)) for i in ids]
            per_run.append(scores([gold[i] for i in ids], y_pred))
        spread[name] = {m: float(np.std([s[m] for s in per_run])) for m in ("precision",
                                                                             "recall", "f1")}
        ok = [r for r in mine if r.error is None]
        call_cli = 1000 * float(np.mean([r.cost_usd or 0.0 for r in ok]))
        call_direct = cost_per_1000(
            [direct_input_tokens(r.input_tokens, overhead=overhead) for r in ok],
            [r.output_tokens for r in ok])
        per_call[name] = {"cli": call_cli, "direct": call_direct}
        cli_cost[name], direct_cost[name] = call_cli * share, call_direct * share
        cli_cost[f"{name} vote3"] = 3 * cli_cost[name]
        direct_cost[f"{name} vote3"] = 3 * direct_cost[name]
        latency[name] = latency_summary([r.latency_s for r in ok])
    return preds, spread, cli_cost, direct_cost, latency, per_call


def pipeline_cost_and_latency(regex_hits: dict[str, bool], eligible: set[str], *,
                              haiku_cost_per_1000: float, regex_latencies: list[float],
                              haiku_latencies: dict[str, float]) -> tuple[float, dict]:
    """Haiku is only called on eligible messages the regex did not flag."""
    ids = sorted(regex_hits)
    sent = [i for i in ids if not regex_hits[i] and i in eligible]
    cost = haiku_cost_per_1000 * len(sent) / len(ids)
    sent_set = set(sent)
    times = [regex_latencies[k] + (haiku_latencies.get(i, 0.0) if i in sent_set else 0.0)
             for k, i in enumerate(ids)]
    return cost, latency_summary(times)


def _opus_error_rate(data: Path) -> tuple[float | None, int]:
    answers_path = data / "answers.json"
    if not answers_path.exists():
        return None, 0
    answers = json.loads(answers_path.read_text())
    opus = {o.msg_id: o.label for o in read_rows(data / "opus.jsonl", OpusLabel)}
    sample = [it.msg_id for it in read_rows(data / "review.jsonl", ReviewItem)
              if it.reason == "random_agreed" and it.msg_id in answers]
    if not sample:
        return None, 0
    return sum(answers[i] != opus[i] for i in sample) / len(sample), len(sample)


def _git_sha(path: Path) -> str:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def evaluate(data: Path, *, finetune: bool = True, finetune_model: str | None = None,
             epochs: int = 6, lr: float = 3e-5):
    messages = load_messages(data)
    gold_rows = read_rows(data / "gold.jsonl", GoldLabel)
    gold = {g.msg_id: g.label for g in gold_rows}
    by_id = {m.id: m for m in messages}
    ids = sorted(i for i in gold if i in by_id)
    gold_rows = [g for g in gold_rows if g.msg_id in by_id]
    eligible = {i for i in ids if by_id[i].llm_eligible}
    folds = _folds(data, gold_rows)

    preds: dict[str, dict[str, bool]] = {}
    probas: dict[str, dict[str, float]] = {}
    latency: dict[str, dict] = {}

    regex_times = []
    preds["regex"] = {}
    for i in ids:
        start = time.perf_counter()
        preds["regex"][i] = regex_predict(by_id[i].text)
        regex_times.append(time.perf_counter() - start)
    latency["regex"] = latency_summary(regex_times)

    overhead = json.loads((data / "haiku_overhead.json").read_text())["overhead_input_tokens"]
    runs = read_rows(data / "haiku_runs.jsonl", HaikuRun)
    h_preds, spread, cli_cost, direct_cost, h_latency, per_call = _haiku(runs, ids, eligible,
                                                                          gold, overhead)
    preds.update(h_preds)
    latency.update(h_latency)
    if "haiku+prior" in preds:
        preds["pipeline"] = {i: preds["regex"][i] or preds["haiku+prior"][i] for i in ids}
        per_msg: dict[str, list[float]] = {}
        for r in runs:
            if r.variant == "with_prior" and r.error is None:
                per_msg.setdefault(r.msg_id, []).append(r.latency_s)
        for name, value in (("direct", per_call["haiku+prior"]["direct"]),
                            ("cli", per_call["haiku+prior"]["cli"])):
            cost, lat = pipeline_cost_and_latency(
                preds["regex"], eligible, haiku_cost_per_1000=value,
                regex_latencies=regex_times,
                haiku_latencies={k: float(np.mean(v)) for k, v in per_msg.items()})
            (direct_cost if name == "direct" else cli_cost)["pipeline"] = cost
            latency["pipeline"] = lat

    subset = [by_id[i] for i in ids]
    for variant, suffix in VARIANTS.items():
        out = tfidf.cv_predict_tfidf(subset, gold_rows, folds, variant=variant)
        preds[f"tfidf{suffix}"] = {i: out[i][0] for i in ids}
        probas[f"tfidf{suffix}"] = {i: out[i][1] for i in ids}
    latency["tfidf"] = latency_summary(tfidf.inference_latency(subset, gold_rows,
                                                               variant="message"))
    devices = []
    if finetune:
        from cdbench.detectors import finetune as ft

        model = finetune_model or ft.DEFAULT_MODEL
        for variant, suffix in VARIANTS.items():
            out = ft.cv_predict_finetune(subset, gold_rows, folds, variant=variant,
                                         model_name=model, epochs=epochs, lr=lr)
            preds[f"finetune{suffix}"] = {i: out[i][0] for i in ids}
            probas[f"finetune{suffix}"] = {i: out[i][1] for i in ids}
        devices = ["mps", "cpu"] if ft.default_device() == "mps" else ["cpu"]
        for device in devices:
            key = "finetune" if device == devices[0] else f"finetune ({device})"
            latency[key] = latency_summary(ft.inference_latency(
                subset, gold_rows, variant="message", model_name=model, epochs=epochs, lr=lr,
                device=device))

    error_rate, sample_size = _opus_error_rate(data)
    error_ci = (list(wilson_interval(round(error_rate * sample_size), sample_size))
                if error_rate is not None else None)
    meta_path = data / "review_meta.json"
    review_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    reviewed_reasons: dict[str, int] = {}
    for it in read_rows(data / "review.jsonl", ReviewItem):
        reviewed_reasons[it.reason] = reviewed_reasons.get(it.reason, 0) + 1
    off_format = {f"haiku{suffix}": sum(r.off_format for r in runs if r.variant == variant)
                  for variant, suffix in VARIANTS.items()}
    resolved: dict[str, int] = {}
    for g in gold_rows:
        resolved[g.resolved_by] = resolved.get(g.resolved_by, 0) + 1
    extra = {
        "counts": {"in_scope": len(ids), "positives": sum(gold[i] for i in ids),
                   "llm_eligible": len(eligible)},
        "resolved_by": resolved,
        "opus_error_rate_random_sample": error_rate,
        "opus_error_rate_ci": error_ci,
        "opus_random_sample_size": sample_size,
        "review": {"disagreements_before_cap": review_meta.get("disagreements_before_cap"),
                   "reviewed_by_reason": reviewed_reasons},
        "haiku_off_format_answers": off_format,
        "haiku_runs_total": {f"haiku{suffix}": sum(r.variant == variant for r in runs)
                             for variant, suffix in VARIANTS.items()},
        "haiku_latency_note": "measured through claude -p with 4 concurrent calls",
        "haiku_run_spread": spread,
        "cost_cli_per_1000": cli_cost,
        "cost_direct_per_1000": direct_cost,
        "latency": latency,
        "price": {"input_per_mtok": HAIKU_PRICE_INPUT_PER_MTOK,
                  "output_per_mtok": HAIKU_PRICE_OUTPUT_PER_MTOK, "source": PRICE_SOURCE,
                  "read_on": PRICE_READ_ON},
    }
    summary = summarize({i: gold[i] for i in ids}, preds, extra)
    meta = {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_sha": _git_sha(Path(__file__).resolve().parents[2]),
        "hermes_source_sha": "56e48122",
        "seed": SEED,
        "folds": len(folds),
        "haiku_models": sorted({r.model for r in runs if r.error is None}),
        "finetune_model": (finetune_model or "almanach/camembert-base") if finetune else None,
        "finetune_epochs": epochs if finetune else None,
        "finetune_lr": lr if finetune else None,
        "devices": devices,
        "platform": f"{platform.system()} {platform.machine()}",
        "versions": {p: version(p) for p in ("torch", "transformers", "scikit-learn", "numpy")},
    }
    (data / "predictions.json").write_text(json.dumps(
        {"gold": {i: gold[i] for i in ids}, "preds": preds, "probas": probas}))
    return summary, meta, probas, {i: gold[i] for i in ids}
