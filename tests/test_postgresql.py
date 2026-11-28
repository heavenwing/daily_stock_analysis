"""Storage integration tests using an isolated schema on a real PostgreSQL server."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select

from src.config import Config, parse_postgresql_url
from src.repositories.backtest_repo import BacktestRepository
from src.repositories.portfolio_repo import DuplicateTradeUidError, PortfolioRepository
from src.repositories.skill_opinion_sample_repo import SkillOpinionSampleRepository
from src.services.scheduled_analysis_claim import claim_scheduled_analysis
from src.storage import (
    AnalysisHistory,
    BacktestResult,
    ChatSessionStateConflict,
    DatabaseManager,
    PortfolioTrade,
    StockDaily,
    get_db,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def postgres_config(tmp_path, monkeypatch):
    dsn = os.getenv("TEST_POSTGRESQL_URL")
    if not dsn:
        pytest.skip("Set TEST_POSTGRESQL_URL to run PostgreSQL integration tests")
    url = parse_postgresql_url(dsn)
    schema = f"dsa_test_{uuid4().hex}"
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f"CREATE SCHEMA {schema}")
        try:
            test_url = url.update_query_dict({"options": f"-csearch_path={schema}"})
            state_path = tmp_path / "stock_analysis.db"
            monkeypatch.setenv("ENV_FILE", str(tmp_path / ".env"))
            monkeypatch.setenv("DATABASE_PATH", str(state_path))
            monkeypatch.setenv("DATABASE_URL", test_url.render_as_string(hide_password=False))
            Config.reset_instance()
            DatabaseManager.reset_instance()
            yield Config.get_instance()
        finally:
            DatabaseManager.reset_instance()
            Config.reset_instance()
            with engine.begin() as connection:
                connection.exec_driver_sql(f"DROP SCHEMA {schema} CASCADE")
    finally:
        engine.dispose()


def _run_concurrently(action):
    ready = Barrier(2, timeout=10)

    def run():
        ready.wait()
        return action()

    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(run) for _ in range(2)]
        return [task.result(timeout=15) for task in tasks]


def test_select_backend_and_count_concurrent_daily_upserts(postgres_config, monkeypatch):
    db = get_db()
    assert db.is_postgresql
    batch = pd.DataFrame([{"date": date(2026, 11, 25), "close": 100.0}])
    assert sorted(_run_concurrently(lambda: db.save_daily_data(batch, "600519", "test"))) == [0, 1]
    with db.get_session() as session:
        row = session.execute(select(StockDaily)).scalar_one()
        assert row.close == 100
        assert row.canonical_id == "sh600519"

    DatabaseManager.reset_instance()
    assert get_db().is_postgresql  # Schema initialization is repeatable.
    monkeypatch.setenv("DATABASE_URL", "")
    Config.reset_instance()
    DatabaseManager.reset_instance()
    db = get_db()
    assert not db.is_postgresql
    with db.get_session() as session:
        assert session.scalar(select(func.count()).select_from(StockDaily)) == 0


def test_web_history_reads_postgresql_after_api_startup(postgres_config, monkeypatch):
    from fastapi.testclient import TestClient

    from api.app import create_app
    from src.services.runtime_scheduler import RUNTIME_SCHEDULER_SUPPRESS_START_ENV

    monkeypatch.setenv(RUNTIME_SCHEDULER_SUPPRESS_START_ENV, "true")
    monkeypatch.setattr("api.app._schedule_stock_index_background_refresh", lambda *_: None)
    monkeypatch.setattr("src.services.name_to_code_resolver.warmup_akshare_cache", lambda: None)
    with TestClient(create_app()) as client:
        db = get_db()
        with db.session_scope() as session:
            session.add(AnalysisHistory(code="600519", query_id="postgres-history"))
        response = client.get("/api/v1/history")
    assert response.status_code == 200
    assert response.json()["items"][0]["query_id"] == "postgres-history"


def test_chat_upsert_and_snapshot_conflict(postgres_config):
    db = get_db()
    db.ensure_chat_session_generation("chat")
    message_id = db.save_conversation_user_turn("chat", "first turn", ["skill-1"])
    snapshot = db.read_chat_session_snapshot("chat")
    assert snapshot["selected_skill_ids"] == ["skill-1"]
    assert db.upsert_conversation_summary("chat", "summary", message_id, 1, 2, source_snapshot=snapshot)
    snapshot = db.read_chat_session_snapshot("chat")
    assert snapshot["summary"]["summary"] == "summary"

    def commit():
        try:
            db.commit_chat_user_turn(snapshot, "follow-up")
        except ChatSessionStateConflict:
            return False
        return True

    assert sorted(_run_concurrently(commit)) == [False, True]
    assert len(db.read_chat_session_snapshot("chat")["messages"]) == 2


def test_schedule_claims_once_without_local_sqlite(postgres_config):
    due = datetime(2026, 11, 26, 10, tzinfo=timezone.utc)

    def claim():
        return claim_scheduled_analysis(postgres_config, SimpleNamespace(), due)

    assert sorted(_run_concurrently(claim)) == [False, True]
    assert claim_scheduled_analysis(postgres_config, SimpleNamespace(), due + timedelta(days=1))
    assert not os.path.exists(postgres_config.database_path)


def test_portfolio_rollback_after_duplicate_trade(postgres_config):
    db = get_db()
    repository = PortfolioRepository(db)
    account = repository.create_account(name="test", broker=None, market="CN", base_currency="CNY")
    trade = {
        "account_id": account.id, "trade_uid": "trade", "symbol": "600519", "market": "CN",
        "currency": "CNY", "trade_date": date(2026, 11, 25), "side": "buy", "quantity": 1,
        "price": 100, "fee": 0, "tax": 0,
    }
    repository.add_trade(**trade)
    with pytest.raises(DuplicateTradeUidError):
        repository.add_trade(**trade)
    repository.add_trade(**(trade | {"trade_uid": "next-trade"}))
    with db.get_session() as session:
        assert session.scalar(select(func.count()).select_from(PortfolioTrade)) == 2


def test_backtest_json_filter_and_skill_sample_dedup(postgres_config):
    db = get_db()
    snapshots = (
        '{"market_phase_summary":{"market":"cn"}}',
        '{"market_phase_summary":{"market":"us"}}',
        "invalid JSON from legacy history",
    )
    with db.session_scope() as session:
        analyses = [AnalysisHistory(code="600519", context_snapshot=snapshot) for snapshot in snapshots]
        session.add_all(analyses)
        session.flush()
        first_id = analyses[0].id
        session.add_all([BacktestResult(code=row.code, analysis_history_id=row.id) for row in analyses])

    rows, total = BacktestRepository(db).get_results_paginated(
        code=None, code_candidates=("600519",), days=None, offset=0, limit=10,
    )
    assert total == 2
    assert {row[4] for row in rows} == {snapshots[0], snapshots[2]}
    samples = SkillOpinionSampleRepository(db)
    sample = {
        "analysis_history_id": first_id, "stock_code": "600519", "skill_id": "test",
        "signal": "buy", "confidence": 0.9, "sample_schema_version": "v1",
    }
    assert samples.insert_missing([sample]) == 1
    assert samples.insert_missing([sample]) == 0
