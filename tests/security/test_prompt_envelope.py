"""SEC-34..36, SEC-40..41: prompt-injection envelope for RAG chat and cluster
labels.

The gating invariant is the *envelope*: retrieved/label text is wrapped in an
untrusted-data section that is preceded by the guardrail system directives.

SEC-40/41 are the GATING probes. They drive the PROMPT-ASSEMBLY seam directly
-- ``RAGChatbot.chat()`` / ``TabClusterer.generate_cluster_label()`` build
(prompt, system) from retrieved content, and a real ``ClaudeCodeLLMProvider``
(pointed at the hermetic ``agent_cli_recorder`` stub, so no live CLI or model
is involved) wraps them into the exact stdin text the subprocess would
receive. Prompt construction needs no live embedding backend -- retrieval is
replaced with a fixed poisoned result / fixed poisoned tab titles -- so these
run, and can fail, in the hermetic test-unit lane. They are ``sec_seam``
(Python-seam exception, see tests/security/README.md) because they import the
prompt-builder classes directly instead of going through HTTP.

SEC-34/35 are live-stack CERTIFICATION probes, not the gate: they exercise the
same contract end-to-end through a running AI Engine (real embeddings,
``/index``, ``/chat``/``/cluster``, a real agent-CLI subprocess). They are
``sec_managed`` AND additionally skip via ``_require_embeddings`` when no
embedding backend is reachable, so in the hermetic CI lane they skip for lack
of an embedder, and in attached mode they skip because they are
``sec_managed`` -- they can never fail CI. Keep them for live-stack
certification against a real host-AI stack; SEC-40/41 are what actually gates
a prompt-injection regression.

SEC-36 is an advisory model-behaviour canary and never gates.

The poison, the poisoned-tab corpus and the envelope assertion itself live in
``fixtures/prompt_envelope.json`` (plan decision 44) rather than inline. SEC-34
and SEC-35 are ``sec_managed`` and auto-skip in attached mode, so a language
port that only runs attached mode never exercises them; extracting the contract
gives the planned ``SEC_BOOT_*_CMD`` runner (and a TS port's own prompt-builder
tests) the same poison and the same assertions to run. SEC-40/41 read the same
shared block on purpose — they are the gating counterparts and must not be able
to drift away from what SEC-34/35 certify.
"""

import asyncio
import os

import pytest

from tests.security import contracts
from tests.security.conftest import switch_llm_provider

pytestmark = [pytest.mark.security]

FIXTURE = "prompt_envelope"
SHARED = contracts.load_fixture(FIXTURE)["shared"]
POISON = SHARED["poison_text"]
POISON_CANARY = SHARED["poison_canary"]
ENVELOPE_CONTRACT_REF = "{shared.envelope_contract}"


def _envelope_contract(reference: str) -> dict:
    """Resolve a fixture reference to the one shared envelope contract."""
    if reference != ENVELOPE_CONTRACT_REF:
        raise contracts.FixtureContractError(
            f"unknown envelope contract reference {reference!r}"
        )
    return SHARED["envelope_contract"]


def _fill(value: str) -> str:
    """Substitute ``{poison_text}`` / ``{poison_canary}`` into a fixture string."""
    return value.replace("{poison_text}", POISON).replace(
        "{poison_canary}", POISON_CANARY
    )


def _documents(spec_input: dict) -> list:
    return [
        {**doc, "content": _fill(doc["content"])} for doc in spec_input["documents"]
    ]


def _poisoned_tabs(spec_input: dict) -> list:
    tabs = spec_input["poisoned_tabs"]
    content = tabs["content_unit"] * tabs["content_repeat"]
    return [
        {
            "url": tabs["url_template"].format(i=i),
            "title": tabs["title_template"].format(i=i),
            "content": content,
        }
        for i in range(tabs["count"])
    ]


def _assert_envelope_holds(stdin: str, contract: dict | None = None) -> None:
    """Shared envelope assertion reused by every probe in this file.

    The assertion itself is ``fixtures/prompt_envelope.json``'s
    ``shared.envelope_contract``: the guardrail directives must be present and
    must precede the user/content block, and the poison may appear only inside
    the data section.
    """
    contracts.assert_text(
        stdin,
        contract if contract is not None else SHARED["envelope_contract"],
        label="prompt envelope",
    )


def _require_embeddings(ai):
    response = ai.post("/embed", token=ai.token, json={"texts": ["probe"]})
    if response.status_code != 200:
        pytest.skip(
            f"embedding backend unavailable ({response.status_code}); "
            "run against a live AI Engine"
        )


@pytest.mark.sec_seam
def test_sec40_rag_chat_prompt_assembly_wraps_retrieved_text(
    agent_cli_recorder, monkeypatch
):
    """SEC-40: hermetic, GATING counterpart to SEC-34.

    Drives ``RAGChatbot.chat()``'s prompt assembly directly against a fixed
    poisoned "retrieval" result (no embedding backend involved -- ``search``
    is overridden), through a real ``ClaudeCodeLLMProvider`` pointed at the
    hermetic ``agent_cli_recorder`` stub, and asserts the exact envelope
    contract SEC-34 checks live. Seam exception (``sec_seam``): imports the
    RAG/provider classes directly instead of going through HTTP. TS porting
    rule: call the TS port's equivalent chat-prompt-builder function(s)
    directly with the same poisoned fixture and apply ``_assert_envelope_holds``
    (or its TS equivalent) to the output.
    """
    from services.ai_engine.app.chatbot.rag import RAGChatbot
    from services.ai_engine.app.core.llm_client import LLMConfig
    from services.ai_engine.app.providers.agent_cli import ClaudeCodeLLMProvider

    spec = contracts.probe_spec(FIXTURE, "SEC-40")
    expect = contracts.expectations(spec, "SEC-40")
    contracts.assert_expect_keys_consumed(
        expect, ["llm_seam_reached", "llm_stdin"], label="SEC-40"
    )
    envelope = _envelope_contract(expect["llm_stdin"])
    fixed_results = [
        {**hit, "content": _fill(hit["content"])}
        for hit in spec["input"]["fixed_search_results"]
    ]

    class _PoisonedSearchRAGChatbot(RAGChatbot):
        """RAG runtime whose retrieval is fixed to the SEC-34 poison fixture."""

        async def search(self, query, session_id=None, top_k=5):
            return fixed_results

    monkeypatch.setenv(
        spec["staging"]["command_env"], str(agent_cli_recorder.command)
    )

    provider = ClaudeCodeLLMProvider(
        LLMConfig(provider=spec["staging"]["provider"], model="")
    )
    rag = _PoisonedSearchRAGChatbot(db_uri="unused://sec40", embedding_dim=4)
    rag.set_llm_client(provider)

    result = asyncio.run(
        rag.chat(spec["input"]["chat_query"], session_id=spec["input"]["session_id"])
    )
    assert expect["llm_seam_reached"] is True
    assert result["answer"], "chat did not reach the LLM seam"

    dumps = agent_cli_recorder.dumps()
    assert dumps, "chat did not reach the LLM seam"
    _assert_envelope_holds(dumps[-1]["stdin"], envelope)


@pytest.mark.sec_seam
def test_sec41_cluster_label_prompt_assembly_carries_envelope(
    agent_cli_recorder, monkeypatch
):
    """SEC-41: hermetic, GATING counterpart to SEC-35.

    Drives ``TabClusterer.generate_cluster_label()``'s prompt assembly
    directly against a cluster of poisoned tab titles (no embeddings needed --
    labeling only reads already-populated ``Tab`` objects), through a real
    ``ClaudeCodeLLMProvider`` pointed at the hermetic ``agent_cli_recorder``
    stub, and asserts the same envelope contract SEC-35 checks live. Seam
    exception (``sec_seam``): imports the clustering/provider classes directly
    instead of going through HTTP. TS porting rule: call the TS port's
    equivalent label-prompt-builder function(s) directly with the same
    poisoned fixture titles and apply the same envelope assertions.
    """
    from services.ai_engine.app.clustering.pipeline import Cluster, Tab, TabClusterer
    from services.ai_engine.app.core.llm_client import LLMConfig
    from services.ai_engine.app.providers.agent_cli import ClaudeCodeLLMProvider

    spec = contracts.probe_spec(FIXTURE, "SEC-41")
    expect = contracts.expectations(spec, "SEC-41")
    contracts.assert_expect_keys_consumed(
        expect, ["llm_seam_reached", "llm_stdin_all_invocations"], label="SEC-41"
    )
    envelope = _envelope_contract(expect["llm_stdin_all_invocations"])

    monkeypatch.setenv(
        spec["staging"]["command_env"], str(agent_cli_recorder.command)
    )

    provider = ClaudeCodeLLMProvider(
        LLMConfig(provider=spec["staging"]["provider"], model="")
    )
    clusterer = TabClusterer()
    clusterer.set_llm_client(provider)

    tabs = [Tab(**entry) for entry in _poisoned_tabs(spec["input"])]
    cluster = Cluster(id=0, tabs=tabs)

    label = asyncio.run(clusterer.generate_cluster_label(cluster))
    assert expect["llm_seam_reached"] is True
    assert label, "cluster labeling did not reach the LLM seam"

    dumps = agent_cli_recorder.dumps()
    assert dumps, "cluster labeling did not reach the LLM seam"
    for dump in dumps:
        _assert_envelope_holds(dump["stdin"], envelope)


@pytest.mark.sec_managed
def test_sec34_rag_chat_wraps_retrieved_text(ai, agent_cli_recorder, monkeypatch):
    """SEC-34: live-stack CERTIFICATION probe, not the gate.

    Requires a reachable embedding backend (``_require_embeddings`` skips
    otherwise) and is ``sec_managed`` (auto-skips in attached mode), so this
    probe can never fail CI by itself -- it certifies the contract end-to-end
    against a real host-AI stack. SEC-40 is the hermetic probe that actually
    gates a prompt-injection regression; keep this one running whenever a live
    stack is available, but do not rely on it to catch a regression in CI.
    """
    spec = contracts.probe_spec(FIXTURE, "SEC-34")
    expect = contracts.expectations(spec, "SEC-34")
    contracts.assert_expect_keys_consumed(
        expect, ["chat_response", "llm_seam_reached", "llm_stdin"], label="SEC-34"
    )
    envelope = _envelope_contract(expect["llm_stdin"])

    assert spec["staging"]["requires_embedding_backend"] is True
    _require_embeddings(ai)
    session = spec["input"]["session_id"]
    index = ai.post(
        "/index",
        token=ai.token,
        json={"session_id": session, "documents": _documents(spec["input"])},
    )
    if index.status_code != 200:
        pytest.skip(f"index unavailable ({index.status_code})")

    monkeypatch.setenv(
        spec["staging"]["command_env"], str(agent_cli_recorder.command)
    )
    switch_llm_provider(ai, spec["staging"]["provider"])
    chat = ai.post(
        "/chat",
        token=ai.token,
        json={"session_id": session, "query": spec["input"]["chat_query"]},
    )
    contracts.assert_response(chat, expect["chat_response"], label="SEC-34 chat")
    assert expect["llm_seam_reached"] is True
    dumps = agent_cli_recorder.dumps()
    assert dumps, "chat did not reach the LLM seam"
    _assert_envelope_holds(dumps[-1]["stdin"], envelope)


@pytest.mark.sec_managed
def test_sec35_cluster_label_prompts_carry_envelope(ai, agent_cli_recorder, monkeypatch):
    """SEC-35: live-stack CERTIFICATION probe, not the gate.

    Requires a reachable embedding backend (``_require_embeddings`` skips
    otherwise) and is ``sec_managed`` (auto-skips in attached mode), so this
    probe can never fail CI by itself -- it certifies the contract end-to-end
    against a real host-AI stack. SEC-41 is the hermetic probe that actually
    gates a prompt-injection regression; keep this one running whenever a live
    stack is available, but do not rely on it to catch a regression in CI.
    """
    spec = contracts.probe_spec(FIXTURE, "SEC-35")
    expect = contracts.expectations(spec, "SEC-35")
    contracts.assert_expect_keys_consumed(
        expect, ["llm_stdin_all_invocations"], label="SEC-35"
    )
    envelope = _envelope_contract(expect["llm_stdin_all_invocations"])

    assert spec["staging"]["requires_embedding_backend"] is True
    _require_embeddings(ai)
    monkeypatch.setenv(
        spec["staging"]["command_env"], str(agent_cli_recorder.command)
    )
    switch_llm_provider(ai, spec["staging"]["provider"])
    urls = _poisoned_tabs(spec["input"])
    response = ai.post(
        "/cluster",
        token=ai.token,
        json={"session_id": spec["input"]["session_id"], "urls": urls},
    )
    if response.status_code != 200:
        pytest.skip(f"clustering unavailable ({response.status_code})")

    dumps = agent_cli_recorder.dumps()
    if not dumps:
        pytest.skip("clustering produced no LLM label call in this runtime")
    for dump in dumps:
        _assert_envelope_holds(dump["stdin"], envelope)


@pytest.mark.integration
@pytest.mark.requires_ollama
@pytest.mark.skipif(
    os.getenv("SEC_ALLOW_NETWORK") != "1", reason="needs live LLM"
)
def test_sec36_model_behavior_canary(ai):
    """Advisory, non-gating: a real model should not echo the injected canary."""
    spec = contracts.probe_spec(FIXTURE, "SEC-36")
    expect = contracts.expectations(spec, "SEC-36")
    contracts.assert_expect_keys_consumed(expect, ["chat_response"], label="SEC-36")

    _require_embeddings(ai)
    session = spec["input"]["session_id"]
    ai.post(
        "/index",
        token=ai.token,
        json={"session_id": session, "documents": _documents(spec["input"])},
    )
    chat = ai.post(
        "/chat",
        token=ai.token,
        json={"session_id": session, "query": spec["input"]["chat_query"]},
    )
    contracts.assert_response(chat, expect["chat_response"], label="SEC-36 chat")
