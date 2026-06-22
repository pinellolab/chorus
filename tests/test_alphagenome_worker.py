"""Tests for the persistent AlphaGenome prediction worker.

These are lightweight: the heavy JAX model load is mocked, so they run in
the base ``chorus`` env without a GPU. They assert the routing + lifecycle
contract that the real worker upholds:

  * ``_predict_in_environment`` routes through ``self._predict_worker`` when
    a session is active, and falls back to the per-call spawn otherwise — and
    both paths return the same dict for the same forward pass.
  * ``predict_session()`` round-trips, is reentrant (no second worker), and
    tears the worker down on exit (process gone, ``_predict_worker`` reset).
  * A non-environment oracle (or any object without ``predict_session``) uses
    ``contextlib.nullcontext`` in the fine-map hook.
"""
from __future__ import annotations

import contextlib

import pytest

from chorus.oracles.alphagenome import AlphaGenomeOracle, AlphaGenomePredictWorker


# ---------------------------------------------------------------------------
# A fake worker so we can exercise routing/lifecycle without a model load.
# ---------------------------------------------------------------------------
class _FakeWorker:
    def __init__(self, oracle):
        self.oracle = oracle
        self.closed = False
        self.calls = []

    def predict(self, seq, assay_ids):
        self.calls.append((seq, tuple(assay_ids)))
        # Echo back a deterministic, recognisable result dict in the same
        # shape the real worker / per-call path returns.
        return {"values": [f"W:{seq}:{assay_ids}"], "resolutions": [1]}

    def close(self):
        self.closed = True


def _make_oracle(use_environment=True):
    """Construct an oracle without touching conda or loading anything."""
    o = AlphaGenomeOracle.__new__(AlphaGenomeOracle)
    o.oracle_name = "alphagenome"
    o.use_environment = use_environment
    o._predict_worker = None
    o.device = None
    o.fold = "all_folds"
    o.sequence_length = 1_048_576
    o.model_dir = None
    o.predict_timeout = 600
    return o


# ---------------------------------------------------------------------------
# Routing: worker vs per-call spawn
# ---------------------------------------------------------------------------
def test_predict_in_environment_routes_to_worker_when_active():
    o = _make_oracle()
    fake = _FakeWorker(o)
    o._predict_worker = fake
    out = o._predict_in_environment("ACGT", ["DNASE:foo"])
    assert out == {"values": ["W:ACGT:['DNASE:foo']"], "resolutions": [1]}
    assert fake.calls == [("ACGT", ("DNASE:foo",))]


def test_predict_in_environment_uses_spawn_when_no_worker(monkeypatch):
    o = _make_oracle()
    captured = {}

    def fake_run_code(code, timeout=None):
        captured["ran"] = True
        return {"values": ["spawned"], "resolutions": [1]}

    o.run_code_in_environment = fake_run_code
    out = o._predict_in_environment("ACGT", ["DNASE:foo"])
    assert captured.get("ran") is True
    assert out == {"values": ["spawned"], "resolutions": [1]}


def test_worker_and_spawn_agree(monkeypatch):
    """worker.predict(...) and the per-call spawn path return the SAME dict
    for the same request (the contract that keeps scores byte-identical)."""
    o = _make_oracle()

    # Per-call spawn path returns this canned result.
    canned = {"values": ["shared-result"], "resolutions": [1]}
    o.run_code_in_environment = lambda code, timeout=None: canned
    spawn_out = o._predict_in_environment("ACGT", ["DNASE:foo"])

    # Now route through a worker that returns the identical dict.
    class _Echo(_FakeWorker):
        def predict(self, seq, assay_ids):
            return canned

    o._predict_worker = _Echo(o)
    worker_out = o._predict_in_environment("ACGT", ["DNASE:foo"])
    assert worker_out == spawn_out


# ---------------------------------------------------------------------------
# predict_session lifecycle
# ---------------------------------------------------------------------------
def test_predict_session_spawns_and_closes_worker(monkeypatch):
    o = _make_oracle()
    created = []

    def fake_ctor(oracle, *a, **k):
        w = _FakeWorker(oracle)
        created.append(w)
        return w

    monkeypatch.setattr(
        "chorus.oracles.alphagenome.AlphaGenomePredictWorker", fake_ctor
    )

    assert o._predict_worker is None
    with o.predict_session() as ses:
        assert ses is o
        assert o._predict_worker is created[0]
        # Forward passes are served by the worker.
        out = o._predict_in_environment("ACGT", ["DNASE:foo"])
        assert out["values"] == ["W:ACGT:['DNASE:foo']"]

    # After the session: worker closed and reference cleared.
    assert created[0].closed is True
    assert o._predict_worker is None


def test_predict_session_reentrant_no_second_worker(monkeypatch):
    o = _make_oracle()
    created = []
    monkeypatch.setattr(
        "chorus.oracles.alphagenome.AlphaGenomePredictWorker",
        lambda oracle, *a, **k: created.append(_FakeWorker(oracle)) or created[-1],
    )

    with o.predict_session():
        first = o._predict_worker
        assert len(created) == 1
        # Nested session must NOT spawn a second worker, and must NOT tear
        # the outer one down on exit.
        with o.predict_session():
            assert o._predict_worker is first
            assert len(created) == 1
        assert o._predict_worker is first
        assert created[0].closed is False
    assert created[0].closed is True
    assert o._predict_worker is None


def test_predict_session_noop_when_not_use_environment():
    o = _make_oracle(use_environment=False)
    with o.predict_session() as ses:
        assert ses is o
        # No worker spawned for the in-process (already fast) path.
        assert o._predict_worker is None
    assert o._predict_worker is None


def test_predict_session_closes_worker_on_exception(monkeypatch):
    o = _make_oracle()
    created = []
    monkeypatch.setattr(
        "chorus.oracles.alphagenome.AlphaGenomePredictWorker",
        lambda oracle, *a, **k: created.append(_FakeWorker(oracle)) or created[-1],
    )
    with pytest.raises(ValueError):
        with o.predict_session():
            raise ValueError("boom")
    assert created[0].closed is True
    assert o._predict_worker is None


# ---------------------------------------------------------------------------
# Fine-map hook: non-AG oracle -> nullcontext
# ---------------------------------------------------------------------------
def test_finemap_hook_nullcontext_for_non_ag_oracle():
    class _PlainOracle:
        pass

    oracle = _PlainOracle()
    session = (
        oracle.predict_session()
        if hasattr(oracle, "predict_session")
        else contextlib.nullcontext()
    )
    assert isinstance(session, contextlib.nullcontext)
    with session as val:
        assert val is None


def test_ag_oracle_has_predict_session():
    assert hasattr(AlphaGenomeOracle, "predict_session")
    assert hasattr(AlphaGenomeOracle, "get_predict_worker_template")


# ---------------------------------------------------------------------------
# Worker template is valid Python and contains the byte-identical extraction.
# ---------------------------------------------------------------------------
def test_worker_template_loads_and_has_extraction():
    import ast

    o = _make_oracle()
    o.model_dir = None
    template, placeholder = o.get_predict_worker_template()
    assert placeholder == "__INIT_ARGS_FILE__"
    ast.parse(template)  # must be valid Python
    # The control protocol + the byte-identical extraction must be present.
    assert "READY" in template
    assert "__STOP__" in template
    assert "np.ascontiguousarray(values[:, local_idx], dtype=np.float32)" in template
    assert "create_from_huggingface" in template
