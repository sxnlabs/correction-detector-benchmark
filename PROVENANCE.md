# Provenance of the production detectors

The regex detector (`src/cdbench/detectors/regex.py`) and the Haiku prompt
(`src/cdbench/detectors/haiku.py`) are frozen copies of the detectors running in
Hermès, the author's private agent fleet:

- source: `lib/feedback_capture.py` (`CORRECTION_PATTERNS`, `IGNORE_PATTERNS`,
  `_normalise`, `is_correction`, `llm_says_correction`), Hermès commit `56e48122`;
- production call path: `lib/llm.py` (`claude -p` with every tool disabled, no MCP
  server, restricted mode, minimal environment). This benchmark uses the same argv
  with `--output-format json` to read the served model, tokens and cost;
- scope constants: `PRIOR_REPLY_WINDOW_SECONDS = 1800`, the previous reply cut to its first 500 characters (`PRIOR_REPLY_EXCERPT_CHARS`) then the last 300 of those in the Haiku prompt,
  `LLM_MIN_CHARS = 6`, messages truncated at 2,000 characters.

Changes made when copying: docstrings and comments quoting real messages were
removed. No pattern contains a proper noun or a client-specific term, so no
pattern was replaced by a placeholder. `tests/test_detectors.py::test_regex_parity_with_hermes`
checks that the copy still matches Hermès when `HERMES_ROOT` points to a checkout.
