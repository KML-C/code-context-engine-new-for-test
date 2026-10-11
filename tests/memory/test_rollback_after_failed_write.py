"""A write that fails and is swallowed must not leave its transaction open.

sqlite3 begins a transaction implicitly before an INSERT and keeps it open
when the statement fails ("database is locked" while another `cce serve`
holds the write lock). On a long-lived connection that pins a stale read
snapshot: later writes fail immediately and the tool still answered "✓".
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp import web

from context_engine.config import Config
from context_engine.integration.mcp_server import ContextEngineMCP
from context_engine.memory import db as memory_db
from context_engine.memory.hooks import add_routes


def _hold_write_lock(db_path: Path) -> sqlite3.Connection:
    peer = memory_db.connect(db_path)
    peer.execute("BEGIN IMMEDIATE")
    return peer


def _peer_commits_a_decision(db_path: Path) -> None:
    peer = memory_db.connect(db_path)
    peer.execute(
        "INSERT INTO decisions (session_id, decision, reason, source, "
        "created_at_epoch, created_at) VALUES (NULL, 'peer', '', 'manual', 0, '')"
    )
    peer.commit()
    peer.close()


@pytest.fixture
def mcp(tmp_path, monkeypatch):
    project_dir = tmp_path / "demo"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    config = Config(
        storage_path=str(tmp_path / "storage"),
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    backend = MagicMock()
    backend._vector_store.count.return_value = 0
    embedder = MagicMock()
    embedder.embed_query = lambda text: [1.0, 0.0]
    server = ContextEngineMCP(
        retriever=MagicMock(), backend=backend, compressor=MagicMock(),
        embedder=embedder, config=config,
    )
    # Fail fast instead of waiting busy_timeout (5 s) on the peer's lock.
    server._memory_conn.execute("PRAGMA busy_timeout = 0")
    yield server
    server._memory_conn.close()


def _db_path(server: ContextEngineMCP) -> Path:
    return Path(memory_db.memory_db_path(server._storage_base))


def test_record_savings_rolls_back_when_locked(tmp_path):
    db_path = tmp_path / "memory.db"
    conn = memory_db.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 0")
    peer = _hold_write_lock(db_path)

    memory_db.record_savings(conn, bucket="grammar", baseline=10, served=5)

    assert not conn.in_transaction
    peer.rollback()
    memory_db.record_savings(conn, bucket="grammar", baseline=10, served=5)
    assert conn.execute("SELECT COUNT(*) FROM savings_log").fetchone()[0] == 1
    peer.close()
    conn.close()


def test_record_decision_reports_and_recovers_after_locked_write(mcp):
    conn = mcp._memory_conn
    peer = _hold_write_lock(_db_path(mcp))

    out = mcp._handle_record_decision({"decision": "first", "reason": "r"})

    assert "⚠" in out[0].text and "✓" not in out[0].text
    assert not conn.in_transaction
    peer.rollback()
    peer.close()

    # A read on this connection, then a commit by another server: with the
    # transaction left open, the read would pin a stale snapshot and every
    # later write here would fail with SQLITE_BUSY.
    conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
    _peer_commits_a_decision(_db_path(mcp))

    out = mcp._handle_record_decision({"decision": "second", "reason": "r"})

    assert out[0].text.startswith("✓ Decision recorded")
    decisions = [r[0] for r in conn.execute("SELECT decision FROM decisions")]
    assert "peer" in decisions and any("second" in d for d in decisions)
    assert not any("first" in d for d in decisions)


def test_record_code_area_reports_and_rolls_back_when_locked(mcp):
    peer = _hold_write_lock(_db_path(mcp))

    out = mcp._handle_record_code_area({"file_path": "src/a.py", "description": "d"})

    assert "⚠" in out[0].text
    assert not mcp._memory_conn.in_transaction
    peer.rollback()
    peer.close()


@pytest.fixture
async def hook_app(tmp_path: Path):
    db_path = tmp_path / "memory.db"
    conn = memory_db.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 0")
    app = web.Application()
    app["memory_db"] = conn
    app["project_name"] = "demo"
    add_routes(app)
    yield app, conn, db_path
    conn.close()


async def test_hook_rolls_back_when_locked_and_recovers(hook_app, aiohttp_client):
    app, conn, db_path = hook_app
    client = await aiohttp_client(app)
    peer = _hold_write_lock(db_path)

    resp = await client.post(
        "/hooks/UserPromptSubmit",
        json={"session_id": "s1", "prompt_number": 1, "prompt_text": "hello"},
    )

    assert resp.status == 202
    assert not conn.in_transaction
    peer.rollback()
    peer.close()

    conn.execute("SELECT COUNT(*) FROM prompts").fetchone()
    _peer_commits_a_decision(db_path)

    resp = await client.post(
        "/hooks/UserPromptSubmit",
        json={"session_id": "s1", "prompt_number": 1, "prompt_text": "hello"},
    )
    assert resp.status == 200
    assert conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0] == 1
