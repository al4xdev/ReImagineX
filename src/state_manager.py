import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.config import load_settings

StateItem = dict[str, Any]
State = list[StateItem]

state_lock = asyncio.Lock()

def get_db_file() -> Path:
    return Path(load_settings().data_dir).expanduser() / "state.json"

def _load() -> State:
    db_file = get_db_file()
    if os.path.exists(db_file):
        with db_file.open("r", encoding="utf-8") as f:
            try:
                value = json.load(f)
                return value if isinstance(value, list) else []
            except (OSError, json.JSONDecodeError):
                return []
    return []

def _save(state: State) -> None:
    db_file = get_db_file()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=db_file.parent, prefix=f".{db_file.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(db_file)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

async def load_state() -> State:
    async with state_lock:
        return _load()

async def save_state(state: State) -> None:
    async with state_lock:
        _save(state)

def delete_item_reparent(item_id: str, state: State) -> tuple[State, list[str]]:
    """
    Deletes the item with item_id, and reparents all its immediate children
    to have their parent_id set to the deleted item's parent_id.
    Returns the new state list and a list containing the deleted item_id.
    """
    parent_id = None
    for item in state:
        if item["id"] == item_id:
            parent_id = item.get("parent_id")
            break
            
    # Reparent immediate children
    for item in state:
        if item.get("parent_id") == item_id:
            item["parent_id"] = parent_id
            
    new_state = [i for i in state if i["id"] != item_id]
    return new_state, [item_id]
