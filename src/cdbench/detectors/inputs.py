"""Model inputs for the two variants."""

from typing import Literal

from cdbench.dataset import Message

Variant = Literal["message", "with_prior"]
SEP = " [SEP] "


def model_input(msg: Message, variant: Variant) -> str:
    if variant == "with_prior" and msg.prior_reply:
        return f"{msg.prior_reply}{SEP}{msg.text}"
    return msg.text
