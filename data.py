import json
import pandas as pd

json_path = "result2.json"
your_name = "Liza Tsoy"
max_messages = 7000

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for msg in data.get("messages", []):
    if msg.get("type") != "message":
        continue
    sender = msg.get("from")
    text = msg.get("text")
    if not text:
        continue

    if isinstance(text, list):
        text = ''.join(part["text"] if isinstance(part, dict) else str(part) for part in text)

    if isinstance(text, str) and text.strip():
        label = 1 if sender == your_name else 0
        rows.append({"text": text.strip(), "label": label})

    if len(rows) >= max_messages:
        break

# === SAVE TO CSV ===
df = pd.DataFrame(rows)
df.to_csv("messages.csv", index=False)
print(f"Saved {len(df)} messages to messages.csv")