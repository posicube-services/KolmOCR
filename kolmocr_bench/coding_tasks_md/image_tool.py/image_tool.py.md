<!-- bbox: [35,22,378,956] -->
python
from PIL import Image
import os
def print_menu():
    print("\n==== Simple Image Tool ====")
    print("1. Resize image")
    print("2. Convert to grayscale")
    print("3. Rotate image")
    print("0. Exit")
    return input("Select: ").strip()
def open_image():
    path = input("Image path: ").strip()
    if not os.path.exists(path):
        print("File not found.")
        return None, None
    try:
        img = Image.open(path)
        return img, path
    except Exception as e:
        print("Failed to open image:", e)
        return None, None
def save_image(img, original_path, suffix):
    base, ext = os.path.splitext(original_path)
    out_path = f"{base}_{suffix}{ext}"
    try:
        img.save(out_path)
        print("Saved to", out_path)
    except Exception as e:
        print("Failed to save:", e)
def resize_image():
    img, path = open_image()
    if img is None:
        return
    try:
        w = int(input("New width: "))
        h = int(input("New height: "))
    except ValueError:
        print("Invalid size.")
        return
    resized = img.resize((w, h))
    save_image(resized, path, f"{w}x{h}")
def to_grayscale():
    img, path = open_image()
    if img is None:
        return
    gray = img.convert("L")
    save_image(gray, path, "gray")
def rotate_image():
    img, path = open_image()
    if img is None:
        return
    try:
        deg = float(input("Degrees: "))
    except ValueError:
        print("Invalid degrees.")
        return
    rotated = img.rotate(deg, expand=True)
    save_image(rotated, path, f"rot{int(deg)}")
def main():
    while True:
        choice = print_menu()
        if choice == "1":
            resize_image()
        elif choice == "2":
            to_grayscale()
        elif choice == "3":
            rotate_image()
        elif choice == "0":
            print("Bye.")
            break
        else:
            print("Unknown menu.")
if __name__ == "__main__":
    main()
