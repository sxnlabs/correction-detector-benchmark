import os
import sys
from datetime import UTC, datetime

import pytest

from cdbench.dataset import Message
from cdbench.detectors.haiku import haiku_predict, haiku_prompt
from cdbench.detectors.pipeline import pipeline_predict
from cdbench.detectors.regex import regex_predict

SYNTHETIC = [
    "non c'est faux", "Non", "non.", "/focus x", "tu as oublie la piece jointe",
    "je t'ai deja dit de ne pas faire ca", "fais le point sur la tresorerie",
    "ARRÊTE de m'envoyer ça", "désormais réponds en français", "c'est pas ça",
    "merci", "ok", "pourquoi tu as envoyé ce mail ?", "relis la conversation",
    "à chaque fois c'est pareil", "tu devrais vérifier avant", "plus jamais ça",
    "peux-tu relancer le build", "n'importe quoi ce résumé", "nope",
    "la prochaine fois préviens-moi", "tu ne sers à rien", "regarde l'historique",
    "quel temps fait-il", "stop les notifications", "jamais le week-end",
    "c'est faux", "je viens de te dire de patienter", "faux, c'est mardi",
    "pas ce que j'ai demandé", "tu dois demander avant", "systématiquement en retard",
    "bonne nuit", "ajoute un todo", "dorénavant en anglais", "à partir de maintenant non",
    "pourquoi t'as fait ça", "tu n'as pas lu", "tout le temps la même erreur", "",
]


def msg(text, prior=None, eligible=True):
    return Message(id="1", ts=datetime(2026, 10, 1, tzinfo=UTC), text=text,
                   prior_reply=prior, llm_eligible=eligible)


@pytest.mark.hermes
def test_regex_parity_with_hermes():
    root = os.environ.get("HERMES_ROOT")
    if not root:
        pytest.skip("HERMES_ROOT not set")
    sys.path.insert(0, os.path.join(root, "lib"))
    from feedback_capture import is_correction

    for text in SYNTHETIC:
        assert regex_predict(text) == is_correction(text), text


def test_haiku_prompt_matches_production_shape():
    prompt = haiku_prompt("m" * 2500, "r" * 400)
    assert "Message : " + "m" * 2000 + "\n" in prompt
    assert "m" * 2001 not in prompt
    assert "Réponse précédente de Hermès : " + "r" * 300 in prompt
    assert "r" * 301 not in prompt
    assert prompt.endswith("Réponds par un seul mot parmi : correction, other.")
    assert "Réponse précédente" not in haiku_prompt("texte", None)


def test_haiku_variant_message_drops_prior(fake_claude, tmp_path):
    fake_claude.set(answer="Correction.")
    call = haiku_predict(msg("tu as oublie", prior="avant"), variant="message", run=0,
                         cache_dir=tmp_path)
    assert call.text == "Correction."
    assert call.key.startswith("haiku|1|message|0|")


def test_haiku_cache_key_changes_with_the_prompt(fake_claude, tmp_path):
    a = haiku_predict(msg("texte", prior="avant"), variant="with_prior", run=0, cache_dir=tmp_path)
    b = haiku_predict(msg("texte", prior="autre"), variant="with_prior", run=0, cache_dir=tmp_path)
    assert a.key != b.key and fake_claude.calls == 2


def test_pipeline_skips_haiku_when_ineligible():
    assert pipeline_predict(msg("ok", eligible=False), haiku_label=True) is False
    assert pipeline_predict(msg("une demande normale"), haiku_label=True) is True
    assert pipeline_predict(msg("non c'est faux"), haiku_label=False) is True
    assert pipeline_predict(msg("une demande normale"), haiku_label=None) is False
