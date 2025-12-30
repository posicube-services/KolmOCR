<!-- bbox: [48,39,952,91] -->
```python
import os
from collections import Counter
```

<!-- bbox: [48,96,952,179] -->
```python
def print_menu():
    print("\n==== Log Analyzer ====")
    print("1.Errors  2.Count  3.Search  0.Exit")
    return input("> ").strip()
```

<!-- bbox: [48,183,952,297] -->
```python
def read_lines(path):
    if not os.path.exists(path):
        print("File not found.")
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()
```

<!-- bbox: [48,302,952,385] -->
```python
def show_errors(lines):
    for line in lines:
        if "ERROR" in line.upper():
            print(line.rstrip())
```

<!-- bbox: [48,390,952,597] -->
```python
def count_levels(lines):
    levels = Counter()
    for line in lines:
        u = line.upper()
        if "ERROR" in u:
            levels["ERROR"] += 1
        elif "WARN" in u:
            levels["WARN"] += 1
        elif "INFO" in u:
            levels["INFO"] += 1
    for k in ("ERROR","WARN","INFO"):
        print(f"{k}: {levels[k]}")
```

<!-- bbox: [48,602,952,700] -->
```python
def search_keyword(lines):
    kw = input("Keyword: ").strip()
    for line in lines:
        if kw and kw in line:
            print(line.rstrip())
```

<!-- bbox: [48,705,952,958] -->
```python
def main():
    path = input("Log file path: ").strip()
    lines = read_lines(path)
    if not lines:
        return
    while True:
        c = print_menu()
        if c == "1": show_errors(lines)
        elif c == "2": count_levels(lines)
        elif c == "3": search_keyword(lines)
        elif c == "0": break
        else: print("Unknown.")

if __name__ == "__main__":
    main()
```
