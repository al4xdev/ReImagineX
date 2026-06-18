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

def delete_item_recursive(item_id: str, state: list) -> tuple[list, list[str]]:
    """
    Recursively deletes an item and all its children/descendants from the state.
    Returns the new state list and a list of all removed IDs.
    """
    children = [i for i in state if i.get("parent_id") == item_id]
    removed_ids = [item_id]
    
    # Recursively collect child IDs and filter state
    for child in children:
        _, child_removed_ids = delete_item_recursive(child["id"], state)
        removed_ids.extend(child_removed_ids)
        
    new_state = [i for i in state if i["id"] not in removed_ids]
    return new_state, removed_ids
