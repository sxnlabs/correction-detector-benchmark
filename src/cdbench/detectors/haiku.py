"""The production Haiku detector, frozen (prompt from Hermes `llm_says_correction`)."""

import hashlib
from pathlib import Path

from cdbench.claude import ClaudeCall, run_claude
from cdbench.dataset import MAX_TEXT_CHARS, PRIOR_REPLY_CHARS, Message
from cdbench.detectors.inputs import Variant


def haiku_prompt(text: str, prior_reply: str | None) -> str:
    context = ""
    if prior_reply:
        context = f"\nRéponse précédente de Hermès : {prior_reply[-PRIOR_REPLY_CHARS:]}"
    return (
        "Nathan écrit à son assistant Hermès sur Telegram. Ce message corrige-t-il "
        "Hermès (reproche, consigne durable du type « ne fais plus X », « toujours Y », "
        "signalement d'une erreur de l'assistant) ? Une demande, une question ou une "
        "info neutre = other.\n\n"
        f"Message : {text.strip()[:MAX_TEXT_CHARS]}{context}\n\n"
        "Réponds par un seul mot parmi : correction, other."
    )


def haiku_label(call: ClaudeCall) -> bool | None:
    """Production reading of the answer; None when the call failed."""
    if call.error is not None or call.text is None:
        return None
    return call.text.strip().lower().startswith("correction")


def haiku_predict(msg: Message, *, variant: Variant, run: int, cache_dir: Path) -> ClaudeCall:
    prior = msg.prior_reply if variant == "with_prior" else None
    prompt = haiku_prompt(msg.text, prior)
    digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]  # a changed prompt is a new call
    return run_claude(prompt, model="haiku", key=f"haiku|{msg.id}|{variant}|{run}|{digest}",
                      cache_dir=cache_dir, timeout=60)
