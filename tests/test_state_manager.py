from pathlib import Path
from typing import Any

import pytest

from src import state_manager


def test_delete_reparents_children() -> None:
    state: list[dict[str, Any]] = [
        {"id": "root", "parent_id": None},
        {"id": "middle", "parent_id": "root"},
        {"id": "child", "parent_id": "middle"},
    ]

    updated, removed = state_manager.delete_item_reparent("middle", state)

    assert removed == ["middle"]
    assert [item["id"] for item in updated] == ["root", "child"]
    assert updated[1]["parent_id"] == "root"


def test_state_round_trip_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    state = [{"id": "root", "parent_id": None}]

    state_manager._save(state)

    assert state_manager._load() == state
    assert not list(tmp_path.glob(".state.json.*"))
