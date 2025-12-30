<!-- bbox: [60,41,940,76] -->
```python
import json, os, random

DATA_FILE = "flashcards.json"
```

<!-- bbox: [60,77,940,171] -->
```python
def load_cards():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
```

<!-- bbox: [60,173,940,208] -->
```python
def save_cards(cards):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
```

<!-- bbox: [60,209,940,256] -->
```python
def menu():
    print("\n== Flashcards ==")
    print("1.List  2.Add  3.Practice  4.WrongOnly  0.Exit")
    return input("> ").strip()
```

<!-- bbox: [60,258,940,364] -->
```python
def list_cards(cards, limit=12):
    if not cards:
        print("No cards.")
        return
    for i, c in enumerate(cards[:limit], 1):
        ok, bad = c.get("correct", 0), c.get("wrong", 0)
        print(f"{i}. {c['front']} -> {c['back']} (ok:{ok}, bad:{bad})")
    if len(cards) > limit:
        print(f"... ({len(cards)-limit} more)")
```

<!-- bbox: [60,365,940,459] -->
```python
def add_card(cards):
    front = input("Front: ").strip()
    back  = input("Back : ").strip()
    if not front or not back:
        print("Both required.")
        return
    cards.append({"front": front, "back": back, "correct": 0, "wrong": 0})
    print("Added.")
```

<!-- bbox: [60,460,940,742] -->
```python
def practice(cards, only_wrong=False):
    if not cards:
        print("No cards.")
        return

    pool = cards
    if only_wrong:
        pool = [c for c in cards if c.get("wrong", 0) > c.get("correct", 0)]
        if not pool:
            print("No wrong-dominant cards.")
            return

    print("Type 'q' to quit.\n")
    while True:
        c = random.choice(pool)
        ans = input(f"Q: {c['front']}  A: ").strip()
        if ans.lower() == "q":
            break
        if ans.lower() == str(c["back"]).strip().lower():
            c["correct"] = c.get("correct", 0) + 1
            print("OK")
        else:
            c["wrong"] = c.get("wrong", 0) + 1
            print(f"NO -> {c['back']}")
```

<!-- bbox: [60,744,940,932] -->
```python
def main():
    cards = load_cards()
    while True:
        ch = menu()
        if ch == "1":
            list_cards(cards)
        elif ch == "2":
            add_card(cards); save_cards(cards)
        elif ch == "3":
            practice(cards); save_cards(cards)
        elif ch == "4":
            practice(cards, only_wrong=True); save_cards(cards)
        elif ch == "0":
            save_cards(cards); print("Bye."); return
        else:
            print("Unknown.")
```

<!-- bbox: [60,933,940,956] -->
```python
if __name__ == "__main__":
    main()
```
