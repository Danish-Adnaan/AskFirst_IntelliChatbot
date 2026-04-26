import json
from datetime import datetime

def load_dataset(path: str = "askfirst_synthetic_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_user(dataset: dict, name: str) -> dict:
    for u in dataset["users"]:
        if u["name"].lower() == name.lower():
            return u
    raise ValueError(f"User '{name}' not found.")

def get_user_names(dataset: dict) -> list:
    return [u["name"] for u in dataset["users"]]

def week_num(ts: str) -> int:
    dt = datetime.fromisoformat(ts)
    start = datetime(2026, 1, 1)
    return ((dt - start).days // 7) + 1

def build_history_context(user: dict) -> str:
    """
    Serialize a user's FULL conversation history into a chronological string.

    CONTEXT MANAGEMENT STRATEGY — Why no chunking:
    Health patterns like telogen effluvium (hair loss 6-12 weeks after nutritional
    deficiency) require the model to hold the ENTIRE timeline. A sliding window or
    retrieval approach would destroy long-lag temporal signals. With ~27 sessions
    averaging ~200 tokens each (~5,400 tokens total), the full history fits easily
    within the 128k context window.
    """
    lines = [
        "=== PATIENT HEALTH RECORD ===",
        f"Name      : {user['name']} | Age: {user['age']} | Gender: {user['gender']}",
        f"Occupation: {user['occupation']} | Location: {user['location']}",
        f"Background: {user['onboarding_notes']}",
        f"Total sessions on record: {len(user['conversations'])}",
        "",
    ]
    for conv in user["conversations"]:
        sn  = conv["session_id"].split("_S")[-1]
        wk  = week_num(conv["timestamp"])
        dt_ = conv["timestamp"][:10]
        tm  = conv["timestamp"][11:16]
        lines.append(f"--- Session {sn} | Week {wk} | {dt_} {tm} | Severity: {conv.get('severity','?')} ---")
        lines.append(f"Patient said     : {conv['user_message']}")
        if conv.get("clary_questions"):
            lines.append(f"Clary asked      : {' / '.join(conv['clary_questions'])}")
        if conv.get("user_followup"):
            lines.append(f"Patient follow-up: {conv['user_followup']}")
        lines.append(f"Clary's response : {conv['clary_response']}")
        lines.append(f"Tags             : {', '.join(conv.get('tags', []))}")
        lines.append("")
    return "\n".join(lines)