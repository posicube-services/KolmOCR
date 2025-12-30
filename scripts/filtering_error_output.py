
import os
import shutil
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

def load_dla_data(dla_dir):
    """
    Loads all JSON files from the DLA directory and builds a dictionary.
    Keys are the document IDs (from the JSON structure).
    """
    dla_map = {}
    json_files = [f for f in os.listdir(dla_dir) if f.endswith('.json')]
    print(f"Loading {len(json_files)} JSON files from {dla_dir}...")
    
    for json_file in tqdm(json_files, desc="Loading DLA JSONs"):
        path = os.path.join(dla_dir, json_file)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                dla_map.update(data)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    return dla_map

def check_image_mean(image_path, threshold=254):
    """
    Returns True if the mean pixel value of the image is >= threshold.
    """
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            mean_val = np.mean(img_array)
            return mean_val >= threshold
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Filter dataset output based on error conditions.")
    parser.add_argument("--input-dir", required=True, help="Directory to filter")
    parser.add_argument("--dla-dir", required=True, help="Directory containing DLA bucket states JSONs")
    parser.add_argument("--translated-mds-dir", default="/my_home/olmOCR-mix-1025-korean/translated_mds", help="Directory containing translated MD files")
    parser.add_argument("--output-dir", required=True, help="Destination directory for valid files")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    dla_dir = os.path.abspath(args.dla_dir)
    translated_mds_dir = os.path.abspath(args.translated_mds_dir)
    output_dir = os.path.abspath(args.output_dir)
    error_dir = os.path.join(output_dir, "error_data")
    
    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create translated_mds output folder
    output_translated_mds_dir = os.path.join(output_dir, "translated_mds")
    if not os.path.exists(output_translated_mds_dir):
        os.makedirs(output_translated_mds_dir)

    error_subdirs = {
        "no_images": os.path.join(error_dir, "no_images_folder"),
        "white_image": os.path.join(error_dir, "white_image")
    }
    
    for path in error_subdirs.values():
        if not os.path.exists(path):
            os.makedirs(path)
        
    print("Loading DLA data...")
    dla_map = load_dla_data(dla_dir)
    print(f"Loaded {len(dla_map)} entries.")
    
    processed_files_count = 0
    copied_valid_count = 0
    copied_error_count = 0
    copied_md_counts = 0
    
    print(f"Scanning {input_dir}...")
    
    for root, dirs, files in tqdm(os.walk(input_dir), desc="Scanning input directory"):
        rel_root = os.path.relpath(root, input_dir)
        if rel_root == ".":
            continue

        folder_name = os.path.basename(root)
        is_doc_dir = rel_root in dla_map
        main_png = os.path.join(root, folder_name + ".png")
        has_main_png = os.path.exists(main_png)
        
        if is_doc_dir or has_main_png:
            doc_id = rel_root
            
            # Helper to copy folder
            def copy_folder(target_base_dir):
                dest_path = os.path.join(target_base_dir, doc_id)
                # dest_parent = os.path.dirname(dest_path) # copytree creates the dest dir, but we need parent? copytree(src, dst)
                # shutil.copytree requires that we provide the full destination directory path.
                # It will create intermediate directories if needed? No, usually just the leaf.
                # Actually shutil.copytree creates the destination directory `dest_path`.
                # We just need to make sure the parent of dest_path exists, or rely on shutil.
                
                try:
                    # shutil.copytree creates the directory 'dest_path'. 
                    # If target_base_dir doesn't exist, it might fail? 
                    # We created target_base_dir (error_subdirs, output_dir) earlier.
                    # But doc_id might be "parent/child", so we need to ensure intermediate dirs exist.
                    
                    # Note: If doc_id mimics the structure (e.g. '00/doc_hash'), then dest_path is 'output_dir/00/doc_hash'.
                    # We need to make sure 'output_dir/00' exists.
                    dest_parent = os.path.dirname(dest_path)
                    if not os.path.exists(dest_parent):
                        os.makedirs(dest_parent)

                    shutil.copytree(root, dest_path, dirs_exist_ok=True)
                    return True
                except Exception as e:
                    print(f"Error copying {root} to {dest_path}: {e}")
                    return False

            # --- Condition 2: Images Folder Missing (if DLA implies it) ---
            # Now we check inside 'root' instead of 'ref_path'
            dla_entry = dla_map.get(doc_id)
            if dla_entry:
                if dla_entry.get("detection_image"):
                    local_images_path = os.path.join(root, "images")
                    if not os.path.isdir(local_images_path):
                        if copy_folder(error_subdirs["no_images"]):
                            copied_error_count += 1
                        dirs[:] = []
                        continue
            
            # --- Condition 3: White Image ---
            if has_main_png:
                if check_image_mean(main_png, threshold=254):
                    if copy_folder(error_subdirs["white_image"]):
                        copied_error_count += 1
                    dirs[:] = []
                    continue
            
            # --- NO CONDITIONS MET: VALID FILE ---
            if copy_folder(output_dir):
                copied_valid_count += 1
                
                # Copy Translated MD
                md_source_path = os.path.join(translated_mds_dir, doc_id + ".md")
                
                if os.path.exists(md_source_path):
                    md_dest_path = os.path.join(output_translated_mds_dir, doc_id + ".md")
                    md_dest_parent = os.path.dirname(md_dest_path)
                    if not os.path.exists(md_dest_parent):
                        os.makedirs(md_dest_parent)
                    try:
                        shutil.copy2(md_source_path, md_dest_path)
                        copied_md_counts += 1
                    except Exception as e:
                        print(f"Error copying MD file {md_source_path}: {e}")
                
            dirs[:] = [] 
            
    print(f"Processing complete.")
    print(f"Copied {copied_valid_count} valid document folders to {output_dir}.")
    print(f"Copied {copied_error_count} error document folders to subdirectories in {error_dir}.")
    print(f"Copied {copied_md_counts} translated MD files.")

if __name__ == "__main__":
    main()
