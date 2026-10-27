"""Haiku cost, at public API list price."""

# Claude Haiku 5.5, prompts up to 100,000 tokens.
HAIKU_PRICE_INPUT_PER_MTOK = 0.10
HAIKU_PRICE_OUTPUT_PER_MTOK = 0.50
PRICE_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICE_READ_ON = "2026-10-26"


def cost_per_1000(input_tokens: list[int], output_tokens: list[int]) -> float:
    """USD for 1,000 messages, from the mean tokens per message."""
    n = len(input_tokens)
    mean_in = sum(input_tokens) / n
    mean_out = sum(output_tokens) / n
    return 1000 * (mean_in * HAIKU_PRICE_INPUT_PER_MTOK
                   + mean_out * HAIKU_PRICE_OUTPUT_PER_MTOK) / 1_000_000


def direct_input_tokens(cli_input_tokens: int, *, overhead: int) -> int:
    """Input tokens a direct API call would send: the CLI total minus its own prompt.

    `overhead` is measured once with a near-empty prompt (see `cdbench evaluate`).
    """
    return max(1, cli_input_tokens - overhead)
