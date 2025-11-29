<!-- bbox: [35,26,520,948] -->
python
import os
from collections import Counter
def print_menu():
    print("\n==== Log Analyzer ====")
    print("1. Show error lines")
    print("2. Count by level (INFO/WARN/ERROR)")
    print("3. Search keyword")
    print("0. Exit")
    return input("Select: ").strip()
def read_lines(path):
    if not os.path.exists(path):
        print("File not found.")
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()
def show_errors(lines):
    print("\n[Error lines]")
    for line in lines:
        if "ERROR" in line.upper():
            print(line.rstrip())
def count_levels(lines):
    levels = Counter()
    for line in lines:
        u = line.upper()
        if "ERROR" in u:
            levels["ERROR"] += 1
        elif "WARN" in u or "WARNING" in u:
            levels["WARN"] += 1
        elif "INFO" in u:
            levels["INFO"] += 1
    print("\n[Level counts]")
    for level in ["ERROR", "WARN", "INFO"]:
        print(f"{level}: {levels[level]}")
def search_keyword(lines):
    kw = input("Keyword: ").strip()
    if not kw:
        print("Keyword required.")
        return
    print(f"\n[Lines containing '{kw}']")
    for line in lines:
        if kw in line:
            print(line.rstrip())
def main():
    path = input("Log file path: ").strip()
    lines = read_lines(path)
    if not lines:
        return
    while True:
        choice = print_menu()
        if choice == "1":
            show_errors(lines)
        elif choice == "2":
            count_levels(lines)
        elif choice == "3":
            search_keyword(lines)
        elif choice == "0":
            print("Bye.")
            break
        else:
            print("Unknown menu.")
if __name__ == "__main__":
    main()
