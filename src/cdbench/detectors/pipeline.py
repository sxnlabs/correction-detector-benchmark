"""The production pipeline: regex first, Haiku for eligible messages it misses."""

from cdbench.dataset import Message
from cdbench.detectors.regex import regex_predict


def pipeline_predict(msg: Message, haiku_label: bool | None) -> bool:
    if regex_predict(msg.text):
        return True
    return bool(msg.llm_eligible and haiku_label)
