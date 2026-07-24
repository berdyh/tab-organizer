"""Frozen black-box security-invariant suite (sec-design 1.0.0).

This package is FROZEN. After merge, any semantic edit to `tests/security/`
requires a decision-log row in `docs/MODULE_INDEX.md` and a bump of
`SECSUITE_VERSION` below. The TypeScript port must pass the same probes by
pointing the harness env vars at its own servers/boot commands.
"""

SECSUITE_VERSION = "1.3.0"
