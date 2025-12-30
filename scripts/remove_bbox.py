import os
import re

target_dir = "output/kolmocr_bench_temp"
# Matches lines that contain only the bbox comment, possibly with whitespace
bbox_line_pattern = re.compile(r'^\s*<!-- bbox: .*? -->\s*(\n|$)', re.MULTILINE)
# Matches bbox comment inline (just in case)
bbox_inline_pattern = re.compile(r'<!-- bbox: .*? -->')

count = 0
for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".md"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # First try to remove whole lines
            new_content = bbox_line_pattern.sub("", content)
            
            # Then remove any remaining inline ones
            new_content = bbox_inline_pattern.sub("", new_content)
            
            if content != new_content:
                count += 1
                print(f"Modifying {file_path}")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

print(f"Modified {count} files.")
