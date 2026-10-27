"""The production regex detector, frozen.

Copied from Hermes `lib/feedback_capture.py` (see PROVENANCE.md for the source
commit). Kept here so the benchmark stays reproducible when Hermes evolves.
"""

import re
import unicodedata

CORRECTION_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"^\s*(non|nan|nope|faux)\b",
    r"\b(jamais|toujours|tout le temps|a chaque fois|systematiquement)\b",
    r"\b(arrete|arretez|stop)\b",
    r"\bpourquoi (tu|t'|hermes)\b",
    r"\b(c'est faux|n'importe quoi|c'est pas ca|ce n'est pas ca|pas ce que j'ai demande)\b",
    r"\btu (me sers|ne sers|ne me sers|dois|devrais|aurais du|n'as pas|as oublie)\b",
    r"\bje (t'ai|viens de te) (deja )?(dit|dire|demande|demander|explique|expliquer)\b",
    r"\b(regarde|relis|lis) (l'historique|la conversation|ce que j'ai)\b",
    r"\bplus jamais\b",
    r"\b(la prochaine fois|desormais|dorenavant|a partir de maintenant)\b",
))

# Messages that look like corrections but are routine answers or commands.
IGNORE_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"^\s*(non|nan|nope)\s*[.!]?\s*$",
    r"^\s*/",
))


def normalise(text: str) -> str:
    """Lowercase, strip accents (NFKD) and unify apostrophes."""
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.replace("’", "'")


def regex_predict(text: str) -> bool:
    """True when the message reads as a correction (production `is_correction`)."""
    if not text or not text.strip():
        return False
    norm = normalise(text)
    if any(p.search(norm) for p in IGNORE_PATTERNS):
        return False
    return any(p.search(norm) for p in CORRECTION_PATTERNS)
