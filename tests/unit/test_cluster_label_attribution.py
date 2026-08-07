"""Spec R5: a generated label carries who generated it.

R5 was deferred to the TypeScript cutover on the grounds that it "needs the
content-addressed schema", and `docs/SPEC-provider-routing.md`'s build table
says to add no columns to the Python store. The same document argues the
opposite two sections earlier: "cheap to add while touching this code;
expensive to retrofit once rows exist without it."

Settled by checking whether rows actually accumulate rather than by weighing
the two claims. They do -- `routes.py` persists ai-engine's cluster payload
verbatim into `sessions.clusters`, a JSON TEXT column. A JSON field is not a
column, so the stamp honours the prohibition, needs no migration, and builds
no part of the content-addressed schema twice.
"""

import pytest

from services.ai_engine.app.clustering.pipeline import Cluster, Tab, TabClusterer


class _StubLLMClient:
    """Minimal stand-in exposing the attribute the stamp reads."""

    def __init__(self, provider="codex_cli", model="gpt-5.6-luna"):
        self.llm_config = type(
            "Cfg", (), {"provider": provider, "model": model}
        )()
        self.calls = 0

    async def generate(self, prompt, system=None):
        self.calls += 1
        return "Rust Async Runtimes"


def _cluster() -> Cluster:
    return Cluster(
        id=0,
        tabs=[
            Tab(url="https://docs.rs/tokio", title="tokio"),
            Tab(url="https://async.rs/", title="async-std"),
        ],
    )


@pytest.mark.asyncio
async def test_a_generated_label_records_provider_model_and_prompt_version():
    clusterer = TabClusterer()
    clusterer.set_llm_client(_StubLLMClient())
    cluster = _cluster()

    name = await clusterer.generate_cluster_label(cluster)

    stamp = cluster.metadata["generated_by"]
    assert name == "Rust Async Runtimes"
    assert stamp["provider"] == "codex_cli"
    assert stamp["model"] == "gpt-5.6-luna"
    assert stamp["prompt_version"] == TabClusterer.LABEL_PROMPT_VERSION
    assert stamp["run_at"]


@pytest.mark.asyncio
async def test_the_stamp_survives_into_the_persisted_payload():
    """The stamp must be PER-CLUSTER, not per-response.

    Backend does `clusters = response.json()["clusters"]` and persists only
    that array, so a top-level attribution field would be dropped on the way
    to storage and the rows would still be unattributable. This asserts on the
    exact structure that reaches `sessions.clusters`.
    """
    clusterer = TabClusterer()
    clusterer.set_llm_client(_StubLLMClient(provider="ollama", model="llama3.2:3b"))
    cluster = _cluster()
    await clusterer.generate_cluster_label(cluster)

    payload = clusterer.to_dict([cluster])

    assert payload[0]["generated_by"]["provider"] == "ollama"
    assert payload[0]["generated_by"]["model"] == "llama3.2:3b"


def test_a_label_nothing_generated_carries_no_attribution():
    """An empty stamp would claim attribution for a name nobody produced.

    Noise points become "Uncategorized" clusters that `cluster()` never sends
    to a provider. Those rows must have no `generated_by` at all rather than
    one with null fields, which would read as "a provider produced this".
    """
    clusterer = TabClusterer()
    cluster = Cluster(id=1, name="Uncategorized", tabs=[Tab(url="https://x.test/")])

    payload = clusterer.to_dict([cluster])

    assert "generated_by" not in payload[0]


@pytest.mark.asyncio
async def test_a_failed_label_is_not_stamped_as_generated():
    """Non-vacuity for the failure path: a placeholder is not a label.

    `UNLABELED_CLUSTER_NAME` means the provider call failed. Stamping that
    with a provider and model would attribute a placeholder to a model that
    never produced it -- the exact confusion the placeholder text exists to
    prevent.
    """

    class _Exploding:
        llm_config = type("Cfg", (), {"provider": "openrouter", "model": "x"})()

        async def generate(self, prompt, system=None):
            raise TimeoutError("provider timed out")

    clusterer = TabClusterer()
    clusterer.set_llm_client(_Exploding())
    cluster = _cluster()

    name = await clusterer.generate_cluster_label(cluster)

    assert "label generation failed" in name
    assert "generated_by" not in cluster.metadata
    assert "generated_by" not in clusterer.to_dict([cluster])[0]
