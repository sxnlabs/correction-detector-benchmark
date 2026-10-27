"""Send the messages to arbitrate to Nathan on Telegram, and collect his answers.

Optional and specific to the author's setup: it relies on Hermes
`lib/question_router.py` (`HERMES_ROOT`). Anyone rerunning the benchmark can
fill `answers.json` ({"<message id>": true|false}) by any other means.

Usage:
  uv run python -m scripts.telegram_review send --data-dir DIR
  uv run python -m scripts.telegram_review collect --data-dir DIR
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cdbench.dataset import load_messages
from cdbench.labeling.arbitration import ReviewItem
from cdbench.store import read_rows

QUESTIONS = "review_questions.jsonl"
AGENT = "correction-benchmark"
OPTIONS = ("Correction", "Autre")
WAIT = timedelta(hours=12)


def _sent(data_dir: Path) -> list[dict]:
    path = data_dir / QUESTIONS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def answer_to_label(answer: str) -> bool | None:
    text = answer.split(":", 1)[1] if answer.startswith("q-") and ":" in answer else answer
    text = text.strip().lower()
    if text.startswith("corr"):
        return True
    if text.startswith(("autre", "other")):
        return False
    return None


def _question_text(msg, number: int, total: int) -> str:
    prior = f"Hermès t'avait répondu :\n« {msg.prior_reply} »\n\nTu as ensuite écrit :\n" \
        if msg.prior_reply else "Tu as écrit à Hermès :\n"
    return (f"Étiquetage {number}/{total}\n\n{prior}« {msg.text} »\n\n"
            "Dans ce message, tu corrigeais Hermès ?\n"
            "Correction = reproche, consigne durable (« ne fais plus X », « toujours Y ») "
            "ou erreur signalée.\n"
            "Autre = demande, question, info, remerciement.")


def send_batch(items: list[ReviewItem], messages: dict, *, data_dir: Path, router,
               batch_size: int = 10, now: datetime | None = None) -> list[str]:
    """Send the next batch once every question already sent has an answer, or 12 hours
    after the oldest unanswered one was sent."""
    now = now or datetime.now(UTC)
    sent = _sent(data_dir)
    pending = [q for q in sent if router.find_answer(q["question_id"]) is None]
    if pending and any(now - datetime.fromisoformat(q["sent_at"]) < WAIT for q in pending):
        return []
    done = {q["msg_id"] for q in sent}
    todo = [it for it in items if it.msg_id not in done][:batch_size]
    new_ids = []
    with (data_dir / QUESTIONS).open("a", encoding="utf-8") as f:
        for offset, it in enumerate(todo, start=len(done) + 1):
            qid = router.ask(_question_text(messages[it.msg_id], offset, len(items)),
                             agent=AGENT, urgency="urgent", options=list(OPTIONS))
            if qid is None:
                break
            f.write(json.dumps({"msg_id": it.msg_id, "question_id": qid,
                                "sent_at": now.isoformat()}) + "\n")
            new_ids.append(qid)
    return new_ids


def collect_answers(data_dir: Path, *, router) -> dict[str, bool]:
    answers = {}
    for q in _sent(data_dir):
        reply = router.find_answer(q["question_id"])
        label = answer_to_label(reply.get("answer", "")) if reply else None
        if label is not None:
            answers[q["msg_id"]] = label
    return answers


def _router():
    sys.path.insert(0, os.path.join(os.environ.get("HERMES_ROOT", str(Path.home() / "Hermes")),
                                    "lib"))
    import question_router

    return question_router


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("send", "collect"))
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    data = Path(args.data_dir).expanduser()
    router = _router()
    if args.action == "send":
        items = read_rows(data / "review.jsonl", ReviewItem)
        messages = {msg.id: msg for msg in load_messages(data)}
        sent = send_batch(items, messages, data_dir=data, router=router)
        print(f"sent: {len(sent)}, total sent: {len(_sent(data))}/{len(items)}")
    else:
        answers = collect_answers(data, router=router)
        (data / "answers.json").write_text(json.dumps(answers, indent=2))
        print(f"answered: {len(answers)}/{len(_sent(data))}")


if __name__ == "__main__":
    main()
