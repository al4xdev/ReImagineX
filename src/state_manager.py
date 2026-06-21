import os
import json
import asyncio
from config import load_settings

state_lock = asyncio.Lock()

def get_db_file() -> str:
    return os.path.join(load_settings().data_dir, "state.json")

def _load() -> list:
    db_file = get_db_file()
    if os.path.exists(db_file):
        with open(db_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return []
    return []

def _save(state: list) -> None:
    db_file = get_db_file()
    os.makedirs(os.path.dirname(db_file), exist_ok=True)
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

async def load_state() -> list:
    async with state_lock:
        return _load()

async def save_state(state: list) -> None:
    async with state_lock:
        _save(state)

def delete_item_reparent(item_id: str, state: list) -> tuple[list, list[str]]:
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
