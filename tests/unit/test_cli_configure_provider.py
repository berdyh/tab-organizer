"""Regression tests for `scripts/cli.py configure-provider` (SPEC-provider-routing.md R6).

The command's whole point is to never write a provider it could not verify.
Every test here mocks the probe boundary (`cli.probe_llm_provider`,
`cli.probe_embedding_provider`, `cli.probe_ollama_installed_models`) rather
than depending on what CLIs/keys/Ollama models happen to be installed on the
machine running the test suite -- see SPEC-provider-routing.md's warning
about tests that cannot fail.
"""

import argparse

import pytest

from scripts import cli, init


def _configure_args(**overrides):
    values = {
        "provider": None,
        "llm_model": None,
        "embedding_provider": None,
        "embedding_model": None,
        "yes": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _write_env(tmp_path, monkeypatch, contents="AI_PROVIDER=openrouter\n"):
    env_file = tmp_path / ".env"
    env_file.write_text(contents)
    monkeypatch.setattr(init, "ENV_FILE", env_file)
    monkeypatch.setattr(cli, "load_env_file", lambda: None)
    return env_file


# --------------------------------------------------------------------------
# Gating test 1: a provider with a missing binary/key is never offered.
# --------------------------------------------------------------------------


def test_unavailable_llm_provider_excluded_from_candidates(monkeypatch):
    from config.config_loader import get_ai_config

    ai_config = get_ai_config()

    def fake_probe(provider):
        if provider == "codex_cli":
            return {"available": False, "reason": "codex is not on PATH"}
        return {"available": True, "reason": None}

    monkeypatch.setattr(cli, "probe_llm_provider", fake_probe)

    names = [name for name, _ in cli._probe_available_llm_providers(ai_config)]

    assert "codex_cli" not in names
    assert "claude_code" in names  # sanity: a real available provider still shows up


# --------------------------------------------------------------------------
# Gating test 2: an embedding provider that cannot embed is never offered,
# even if a (mocked) probe would call it available.
#
# `_probe_available_embedding_providers` applies two catalog filters before
# ever probing: `is_provider_supported(provider, "embeddings")` and
# `get_provider_models(provider, "embedding")`. In the live catalog they
# currently agree on every provider (openrouter has supports.embeddings=false
# AND zero listed embedding models), so this test alone cannot tell you which
# filter did the excluding -- delete either one and this test still passes,
# because the other still catches openrouter. The two tests below isolate
# each filter with a synthetic disagreement so each can fail on its own.
# --------------------------------------------------------------------------


def test_openrouter_never_offered_for_embeddings_even_if_probe_says_available(monkeypatch):
    from config.config_loader import get_ai_config

    ai_config = get_ai_config()

    # Deliberately permissive probe: everything "available". Real catalog
    # values for both filters agree that openrouter is excluded -- this is a
    # smoke test that the combined path behaves correctly end to end, not
    # proof of which filter is responsible (see the two tests below for that).
    monkeypatch.setattr(cli, "probe_embedding_provider", lambda provider: {"available": True})

    names = [name for name, _ in cli._probe_available_embedding_providers(ai_config)]

    assert "openrouter" not in names
    assert "ollama" in names


def test_supports_embeddings_filter_excludes_provider_even_with_phantom_models(monkeypatch):
    """Isolates `is_provider_supported(provider, "embeddings")` from the
    `get_provider_models` filter it is redundant with today.

    Reproduces the historical regression this catalog exists to prevent: the
    catalog briefly listed three embedding model IDs for openrouter that did
    not exist (verified 2026-08-04, none of the 338 catalogued models has an
    embedding modality). Simulates that here by making `get_provider_models`
    report a model for openrouter while leaving `is_provider_supported`
    untouched (real catalog value: False). If the `is_provider_supported`
    filter were ever deleted, this is exactly the scenario that would let
    openrouter through.
    """
    from config.config_loader import get_ai_config

    ai_config = get_ai_config()
    real_get_provider_models = ai_config.get_provider_models

    def phantom_models(provider, model_type=None):
        if provider == "openrouter" and model_type == "embedding":
            return ["phantom/fake-embed-1"]
        return real_get_provider_models(provider, model_type)

    monkeypatch.setattr(ai_config, "get_provider_models", phantom_models)
    monkeypatch.setattr(cli, "probe_embedding_provider", lambda provider: {"available": True})

    names = [name for name, _ in cli._probe_available_embedding_providers(ai_config)]

    assert "openrouter" not in names, (
        "is_provider_supported(embeddings) must exclude a provider even when "
        "get_provider_models reports models for it"
    )


def test_embedding_models_filter_excludes_provider_even_with_supports_flag_true(monkeypatch):
    """Isolates `get_provider_models` from the `is_provider_supported` filter
    it is redundant with today.

    Simulates a catalog entry whose `supports.embeddings` flag is misconfigured
    True but which lists no actual embedding models, leaving
    `get_provider_models` untouched (real catalog value: empty list for
    openrouter). If the `get_provider_models` filter were ever deleted, this
    is exactly the scenario that would let such a provider through on the
    strength of the flag alone.
    """
    from config.config_loader import get_ai_config

    ai_config = get_ai_config()
    real_is_provider_supported = ai_config.is_provider_supported

    def flag_says_yes(provider, capability):
        if provider == "openrouter" and capability == "embeddings":
            return True
        return real_is_provider_supported(provider, capability)

    monkeypatch.setattr(ai_config, "is_provider_supported", flag_says_yes)
    monkeypatch.setattr(cli, "probe_embedding_provider", lambda provider: {"available": True})

    names = [name for name, _ in cli._probe_available_embedding_providers(ai_config)]

    assert "openrouter" not in names, (
        "get_provider_models must exclude a provider even when "
        "is_provider_supported(embeddings) is (misconfigured) True"
    )


# --------------------------------------------------------------------------
# Gating test 3: the command refuses to write a provider whose probe failed.
# --------------------------------------------------------------------------


def test_refuses_explicit_provider_that_fails_its_probe(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path, monkeypatch)

    # codex_cli/codex_acp probe available so the command doesn't bail out on
    # "nothing usable" -- this isolates the explicit-provider refusal path.
    monkeypatch.setattr(
        cli,
        "probe_llm_provider",
        lambda provider: {"available": provider != "claude_code", "reason": "not logged in"},
    )

    args = _configure_args(provider="claude_code")

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_configure_provider(args)

    message = str(exc_info.value)
    assert "claude_code" in message
    assert "failed its availability probe" in message
    assert env_file.read_text() == "AI_PROVIDER=openrouter\n"


def test_refuses_when_nothing_is_verified_available(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path, monkeypatch)

    monkeypatch.setattr(
        cli, "probe_llm_provider", lambda provider: {"available": False, "reason": "not found"}
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_configure_provider(_configure_args())

    assert "No LLM provider is verified available" in str(exc_info.value)
    assert env_file.read_text() == "AI_PROVIDER=openrouter\n"


# --------------------------------------------------------------------------
# Gating test 4: EMBEDDING_DIMENSIONS is left blank, not written.
# --------------------------------------------------------------------------


def test_configure_provider_writes_verified_choice_and_blanks_dimensions(tmp_path, monkeypatch):
    env_file = _write_env(
        tmp_path,
        monkeypatch,
        contents="\n".join(
            [
                "AI_PROVIDER=openrouter",
                "EMBEDDING_PROVIDER=openrouter",
                "LLM_MODEL=",
                "EMBEDDING_MODEL=",
                "EMBEDDING_DIMENSIONS=999",
            ]
        )
        + "\n",
    )

    # Every LLM/embedding provider probes available. Provider selection is
    # given explicitly (--provider/--embedding-provider ARE the deliberate
    # choice under non-interactive use -- see the tty-gate tests below);
    # model selection is left to prompt_choice()'s non-tty fallback, which
    # deterministically picks the catalog default for the chosen provider.
    monkeypatch.setattr(cli, "probe_llm_provider", lambda provider: {"available": True})
    monkeypatch.setattr(cli, "probe_embedding_provider", lambda provider: {"available": True})
    monkeypatch.setattr(
        cli,
        "probe_ollama_installed_models",
        lambda base_url, timeout=2.0: {"nomic-embed-text"},
    )

    cli.cmd_configure_provider(
        _configure_args(provider="claude_code", embedding_provider="ollama")
    )

    env_text = env_file.read_text()
    assert "AI_PROVIDER=claude_code" in env_text
    assert "EMBEDDING_PROVIDER=ollama" in env_text
    assert "EMBEDDING_MODEL=nomic-embed-text" in env_text
    assert "EMBEDDING_DIMENSIONS=999" not in env_text
    assert "EMBEDDING_DIMENSIONS=\n" in env_text or env_text.rstrip().endswith(
        "EMBEDDING_DIMENSIONS="
    )


# --------------------------------------------------------------------------
# Gating test 5: a non-interactive run with no explicit --provider must never
# silently auto-select one. AI_PROVIDER/EMBEDDING_PROVIDER in .env is read
# elsewhere (R1, R3) as proof a human deliberately chose it -- an unattended
# auto-pick would forge that record of consent.
# --------------------------------------------------------------------------


def test_refuses_to_auto_select_llm_provider_when_noninteractive(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path, monkeypatch)
    assert not cli.sys.stdin.isatty()  # sanity: this is exactly the scenario under test

    # Every downstream step is mocked to succeed, so that if the tty gate is
    # ever removed, this test proves the forged value actually reaches .env
    # (a changed exit code alone would not be convincing).
    monkeypatch.setattr(cli, "probe_llm_provider", lambda provider: {"available": True})
    monkeypatch.setattr(cli, "probe_embedding_provider", lambda provider: {"available": True})
    monkeypatch.setattr(
        cli, "probe_ollama_installed_models", lambda base_url, timeout=2.0: {"nomic-embed-text"}
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_configure_provider(_configure_args())

    message = str(exc_info.value)
    assert "will not choose a provider on your behalf" in message
    assert "--provider" in message
    # Lists what it found, so a scripted caller learns valid values without a
    # second run.
    assert "claude_code" in message
    assert env_file.read_text() == "AI_PROVIDER=openrouter\n"


def test_refuses_to_auto_select_embedding_provider_when_noninteractive(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path, monkeypatch)

    # LLM provider given explicitly (the legitimate non-interactive route) so
    # this isolates the embedding-provider gate specifically.
    monkeypatch.setattr(cli, "probe_llm_provider", lambda provider: {"available": True})
    monkeypatch.setattr(cli, "probe_embedding_provider", lambda provider: {"available": True})

    args = _configure_args(provider="claude_code", llm_model="sonnet")

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_configure_provider(args)

    message = str(exc_info.value)
    assert "will not choose a provider on your behalf" in message
    assert "--embedding-provider" in message
    assert "ollama" in message
    # Nothing written at all -- not even the LLM provider resolved earlier in
    # the same run.
    assert env_file.read_text() == "AI_PROVIDER=openrouter\n"


def test_explicit_provider_flags_still_work_noninteractively(tmp_path, monkeypatch):
    """The legitimate scripted route: --provider/--embedding-provider ARE the
    deliberate choice, so this must keep working with no tty at all."""
    env_file = _write_env(tmp_path, monkeypatch)

    monkeypatch.setattr(cli, "probe_llm_provider", lambda provider: {"available": True})
    monkeypatch.setattr(cli, "probe_embedding_provider", lambda provider: {"available": True})
    monkeypatch.setattr(
        cli, "probe_ollama_installed_models", lambda base_url, timeout=2.0: {"nomic-embed-text"}
    )

    args = _configure_args(provider="claude_code", embedding_provider="ollama")
    cli.cmd_configure_provider(args)

    env_text = env_file.read_text()
    assert "AI_PROVIDER=claude_code" in env_text
    assert "EMBEDDING_PROVIDER=ollama" in env_text


# --------------------------------------------------------------------------
# Ollama special case (WI0-B1): refuse an unpulled model rather than writing it.
# --------------------------------------------------------------------------


def test_refuses_to_write_unpulled_ollama_model(tmp_path, monkeypatch):
    env_file = _write_env(tmp_path, monkeypatch)

    monkeypatch.setattr(
        cli, "probe_llm_provider", lambda provider: {"available": provider == "ollama"}
    )
    monkeypatch.setattr(
        cli, "probe_ollama_installed_models", lambda base_url, timeout=2.0: set()
    )

    args = _configure_args(provider="ollama", llm_model="qwen3")

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_configure_provider(args)

    message = str(exc_info.value).lower()
    assert "unpulled" in message
    assert "AI_PROVIDER=ollama" not in env_file.read_text()


def test_pulls_ollama_model_when_confirmed_with_yes_flag(tmp_path, monkeypatch):
    """--yes offers to pull, and only proceeds once the pull is verified."""
    env_file = _write_env(tmp_path, monkeypatch)

    monkeypatch.setattr(
        cli, "probe_llm_provider", lambda provider: {"available": provider == "ollama"}
    )
    monkeypatch.setattr(cli, "probe_embedding_provider", lambda provider: {"available": True})

    pulled = []
    installed = {"nomic-embed-text"}  # embedding model already present

    def fake_installed(base_url, timeout=2.0):
        return set(installed)

    def fake_pull(base_url, model, timeout=1800.0):
        pulled.append(model)
        installed.add(model)

    monkeypatch.setattr(cli, "probe_ollama_installed_models", fake_installed)
    monkeypatch.setattr(cli, "pull_ollama_model", fake_pull)

    args = _configure_args(
        provider="ollama", llm_model="qwen3", embedding_provider="ollama", yes=True
    )

    cli.cmd_configure_provider(args)

    assert pulled == ["qwen3"]
    env_text = env_file.read_text()
    assert "AI_PROVIDER=ollama" in env_text
    assert "LLM_MODEL=qwen3" in env_text


# --------------------------------------------------------------------------
# Gating test: the five .env keys land in ONE read-modify-write pass, not
# four/five sequential ones, so an IO failure between them cannot leave .env
# with only some of the new selection applied. scripts/MODULE.md calls
# partial writes a hard constraint.
# --------------------------------------------------------------------------


def test_configure_provider_writes_all_env_keys_in_a_single_pass(tmp_path, monkeypatch):
    env_file = _write_env(
        tmp_path,
        monkeypatch,
        contents="\n".join(
            [
                "AI_PROVIDER=openrouter",
                "EMBEDDING_PROVIDER=openrouter",
                "LLM_MODEL=",
                "EMBEDDING_MODEL=",
                "EMBEDDING_DIMENSIONS=999",
            ]
        )
        + "\n",
    )

    monkeypatch.setattr(cli, "probe_llm_provider", lambda provider: {"available": True})
    monkeypatch.setattr(cli, "probe_embedding_provider", lambda provider: {"available": True})
    monkeypatch.setattr(
        cli,
        "probe_ollama_installed_models",
        lambda base_url, timeout=2.0: {"nomic-embed-text"},
    )

    calls = []
    real_update_env_vars = init.update_env_vars

    def spying_update_env_vars(pairs):
        calls.append(dict(pairs))
        return real_update_env_vars(pairs)

    # cmd_configure_provider does `from scripts.init import ... update_env_vars`
    # inside the function body, so it re-resolves scripts.init.update_env_vars
    # at call time -- patch the name on init, not on cli, for the spy to take.
    monkeypatch.setattr(init, "update_env_vars", spying_update_env_vars)

    cli.cmd_configure_provider(
        _configure_args(provider="claude_code", embedding_provider="ollama")
    )

    # Exactly one write call, carrying every key together -- not one call per
    # key. A caller that instead did N sequential update_env_var() calls would
    # make this list longer than 1, and an IO failure between those calls
    # could leave .env with only some of them applied.
    assert len(calls) == 1, (
        f"expected configure-provider to batch all keys into one "
        f"update_env_vars() call, got {len(calls)} separate calls: {calls}"
    )
    assert calls[0] == {
        "AI_PROVIDER": "claude_code",
        "LLM_MODEL": "sonnet",
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "nomic-embed-text",
        "EMBEDDING_DIMENSIONS": "",
    }

    env_text = env_file.read_text()
    assert "AI_PROVIDER=claude_code" in env_text
    assert "EMBEDDING_DIMENSIONS=999" not in env_text


def test_ensure_env_file_creates_env_from_template_when_missing(tmp_path, monkeypatch):
    """Hard constraint: never silently write a partial .env -- copy the template
    and say so, exactly like `scripts/init.py` already does."""
    env_file = tmp_path / ".env"
    template_file = tmp_path / ".env.example"
    template_file.write_text("AI_PROVIDER=\nEMBEDDING_PROVIDER=\n")
    monkeypatch.setattr(init, "ENV_FILE", env_file)
    monkeypatch.setattr(init, "ENV_TEMPLATE", template_file)
    monkeypatch.setattr(cli, "load_env_file", lambda: None)

    monkeypatch.setattr(
        cli, "probe_llm_provider", lambda provider: {"available": False, "reason": "not found"}
    )

    with pytest.raises(SystemExit):
        cli.cmd_configure_provider(_configure_args())

    assert env_file.exists()
    assert env_file.read_text() == template_file.read_text()
