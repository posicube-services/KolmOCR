<!-- bbox: [64,43,936,73] -->
```python
from PIL import Image
import os
```

<!-- bbox: [64,75,936,134] -->
```python
def menu():
    print("\n== Simple Image Tool ==")
    print("1.Resize  2.Grayscale  3.Rotate  0.Exit")
    return input("> ").strip()
```

<!-- bbox: [64,136,936,210] -->
```python
def open_img():
    path = input("Image path: ").strip()
    if not os.path.exists(path):
        print("Not found.")
        return None, None
```

<!-- bbox: [64,212,936,286] -->
```python
try:
        return Image.open(path), path
    except Exception as e:
        print("Open failed:", e)
        return None, None
```

<!-- bbox: [64,289,936,408] -->
```python
def save_img(img, src, suffix):
    base, ext = os.path.splitext(src)
    out = f"{base}_{suffix}{ext}"
    try:
        img.save(out)
        print("Saved:", out)
    except Exception as e:
        print("Save failed:", e)
```

<!-- bbox: [64,409,936,558] -->
```python
def resize_img():
    img, path = open_img()
    if img is None: return
    try:
        w = int(input("Width : "))
        h = int(input("Height: "))
    except ValueError:
        print("Invalid size.")
        return
    save_img(img.resize((w, h)), path, f"{w}x{h}")
```

<!-- bbox: [64,560,936,619] -->
```python
def grayscale():
    img, path = open_img()
    if img is None: return
    save_img(img.convert("L"), path, "gray")
```

<!-- bbox: [64,621,936,755] -->
```python
def rotate_img():
    img, path = open_img()
    if img is None: return
    try:
        deg = float(input("Degrees: "))
    except ValueError:
        print("Invalid degrees.")
        return
    save_img(img.rotate(deg, expand=True), path, f"rot{int(deg)}")
```

<!-- bbox: [64,757,936,876] -->
```python
def main():
    while True:
        c = menu()
        if c == "1": resize_img()
        elif c == "2": grayscale()
        elif c == "3": rotate_img()
        elif c == "0": print("Bye."); break
        else: print("Unknown.")
```

<!-- bbox: [64,878,936,908] -->
```python
if __name__ == "__main__":
    main()
```
