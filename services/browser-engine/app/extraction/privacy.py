"""Cheap content checks that run BEFORE anything is handed to an embedder.

Plan decision 37 requires that an authenticated capture never reaches a
non-local embedding provider. Its stated prerequisite -- an auth flag carried
from capture to index -- was built, but the flag it produced (`auth_used`) means
"we spent a credential from our own store", which is structurally blind to the
case that matters most here: a tab open in the user's already-logged-in browser,
where no stored credential is ever spent. So the gate had a signal that could
never fire on the main path.

This module produces the signal the CDP path CAN produce, from content the
harvester already holds, with no extra network calls:

* `classify_privacy` -- does this page look like content behind a login? The
  inverse of the auth-wall check: a sign-in PROMPT means we are logged OUT (that
  page is skipped elsewhere), while sign-OUT affordances, account chrome, and
  OAuth session markers mean we are logged IN and looking at private content.
* `scan_for_secrets` -- does the text carry a credential? A page that merely
  displays an API key must not be shipped to a third-party embedding service,
  and the tabs most likely to contain one (a gist, an IDE, a dashboard) are
  exactly the tabs a real browser has open.

Both are heuristics and are treated as such: a hit HOLDS a document for a human
decision rather than silently deciding for them, and every hit names the signal
that fired so the judgement can be checked. Neither ever echoes a matched
credential value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# "We are logged in" signals
# --------------------------------------------------------------------------

# Sign-OUT affordances. Their presence is the cleanest evidence that a session
# exists: a logged-out page has nothing to log out of. Anchored to avoid
# matching prose like "signout is not supported".
SIGNED_IN_MARKERS = (
    r"\bsign[\s_-]?out\b",
    r"\blog[\s_-]?out\b",
    r"\bmy\s+account\b",
    r"\baccount\s+settings\b",
    r"\bmanage\s+(?:your\s+)?subscription\b",
    r'href=["\'][^"\']*/(?:logout|signout|sign-out|log-out)\b',
    r'\bdata-testid=["\'][^"\']*(?:user-menu|account-menu|avatar)',
)

# OAuth/session infrastructure referenced by an already-authenticated page.
# The user named this one directly: "if the page has google or other requests".
SESSION_INFRASTRUCTURE = (
    r"accounts\.google\.com",
    r"oauth2?\b",
    r"\.auth0\.com",
    r"login\.microsoftonline\.com",
    r"github\.com/login/oauth",
    r"\bsession[_-]?token\b",
    r"\bcsrf[_-]?token\b",
)

# Hosts whose content is private by nature even when the markers are thin --
# a personal workspace, an IDE, an inbox. Kept SHORT and obvious on purpose:
# this is a nudge for the classifier, never an allowlist, and a domain absent
# from it is still classified on its content.
PRIVATE_BY_NATURE_HINTS = (
    "mail.",
    "inbox.",
    "app.",
    "my.",
    "dashboard.",
    "console.",
    "admin.",
)


@dataclass(frozen=True)
class PrivacySignal:
    """Whether a page looks like private, logged-in content, and why."""

    is_private: bool
    confidence: float = 0.0
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_private": self.is_private,
            "confidence": round(self.confidence, 2),
            "reasons": list(self.reasons),
        }


def classify_privacy(html: str, url: str) -> PrivacySignal:
    """Classify a page as private (logged-in) or public, from content alone.

    Deliberately conservative in ONE direction: a page with no signals is
    reported public, because over-flagging turns the hold list into noise
    nobody reads, and the auth-wall skip already covers the logged-out case.
    Every signal that fired is named, so a wrong call is visible rather than
    mysterious.
    """
    haystack = html or ""
    reasons: list[str] = []
    score = 0.0

    for pattern in SIGNED_IN_MARKERS:
        if re.search(pattern, haystack, re.IGNORECASE):
            reasons.append(f"signed_in_marker:{pattern}")
            score += 0.45
            break

    for pattern in SESSION_INFRASTRUCTURE:
        if re.search(pattern, haystack, re.IGNORECASE):
            reasons.append(f"session_infrastructure:{pattern}")
            score += 0.3
            break

    host = (urlparse(url).hostname or "").lower()
    if any(host.startswith(hint) for hint in PRIVATE_BY_NATURE_HINTS):
        reasons.append(f"private_host_shape:{host}")
        score += 0.25

    score = min(score, 1.0)
    # One weak signal alone is not enough; the host shape or a lone OAuth
    # reference appears on plenty of public marketing pages.
    return PrivacySignal(
        is_private=score >= 0.45, confidence=score, reasons=tuple(reasons)
    )


# --------------------------------------------------------------------------
# Secret detection
# --------------------------------------------------------------------------

# High-confidence, vendor-anchored shapes only. A generic "long random string"
# rule would fire on minified JS and every cache-busting hash on the internet,
# and a detector that cries wolf gets switched off -- which is worse than not
# having one.
SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("anthropic_api_key", r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),
    ("openai_api_key", r"\bsk-(?!ant-)[A-Za-z0-9]{32,}"),
    ("openrouter_api_key", r"\bsk-or-[A-Za-z0-9_\-]{20,}"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    ("aws_access_key_id", r"\bAKIA[0-9A-Z]{16}\b"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    ("slack_token", r"\bxox[abprs]-[0-9A-Za-z\-]{10,}"),
    ("private_key_block", r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"),
    ("jwt", r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
)


@dataclass(frozen=True)
class SecretScan:
    """Which secret shapes were found. Never carries a matched VALUE."""

    found: bool = False
    kinds: tuple[str, ...] = ()
    counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "kinds": list(self.kinds),
            "counts": dict(self.counts),
        }


def scan_for_secrets(text: str) -> SecretScan:
    """Report which credential shapes appear in this text, and how many times.

    The matched substring is NEVER returned, logged, or stored. A detector that
    reports "found an Anthropic key" is actionable; one that echoes the key has
    copied a credential into a database, a log file, and a job payload in order
    to warn you that a credential was somewhere it should not be.
    """
    if not text:
        return SecretScan()

    kinds: list[str] = []
    counts: dict[str, int] = {}
    for name, pattern in SECRET_PATTERNS:
        hits = len(re.findall(pattern, text))
        if hits:
            kinds.append(name)
            counts[name] = hits
    return SecretScan(found=bool(kinds), kinds=tuple(kinds), counts=counts)
