<!-- bbox: [35,20,786,961] -->
python
import json
import os
import random
DATA_FILE = "flashcards.json"
def load_cards():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
def save_cards(cards):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
def print_menu():
    print("\n==== Flashcard Trainer ====")
    print("1. List cards")
    print("2. Add card")
    print("3. Practice (random)")
    print("4. Practice wrong ones only")
    print("0. Exit")
    return input("Select: ").strip()
def list_cards(cards):
    if not cards:
        print("No cards.")
        return
    for i, c in enumerate(cards, start=1):
        print(f"{i}. {c['front']} -> {c['back']} (correct: {c.get('correct', 0)}, wrong: {c.get('wrong', 0)})")
def add_card(cards):
    front = input("Front (question): ").strip()
    back = input("Back (answer): ").strip()
    if not front or not back:
        print("Both sides are required.")
        return
    cards.append({"front": front, "back": back, "correct": 0, "wrong": 0})
    print("Card added.")
def practice(cards, only_wrong=False):
    if not cards:
        print("No cards to practice.")
        return
    pool = cards
    if only_wrong:
        pool = [c for c in cards if c.get("wrong", 0) > c.get("correct", 0)]
        if not pool:
            print("No cards with more wrong answers than correct ones.")
            return
    print("Enter 'q' to quit practice.\n")
    while True:
        card = random.choice(pool)
        print(f"Q: {card['front']}")
        ans = input("Your answer: ").strip()
        if ans.lower() == "q":
            break
        if ans.lower() == card["back"].lower():
            print("✅ Correct!")
            card["correct"] = card.get("correct", 0) + 1
        else:
            print(f"❌ Wrong. Correct answer: {card['back']}")
            card["wrong"] = card.get("wrong", 0) + 1
def main():
    cards = load_cards()
    while True:
        choice = print_menu()
        if choice == "1":
            list_cards(cards)
        elif choice == "2":
            add_card(cards)
            save_cards(cards)
        elif choice == "3":
            practice(cards, only_wrong=False)
            save_cards(cards)
        elif choice == "4":
            practice(cards, only_wrong=True)
            save_cards(cards)
        elif choice == "0":
            save_cards(cards)
            print("Bye.")
            break
        else:
            print("Unknown menu.")
if __name__ == "__main__":
    main()
