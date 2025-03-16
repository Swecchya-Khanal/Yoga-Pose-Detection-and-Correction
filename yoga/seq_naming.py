import os
import hashlib

def calculate_hash(image_path):
    """
    Calculate the MD5 hash of an image file.
    Args:
        image_path: Path to the image file.
    Returns:
        Hash string for the image.
    """
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def rename_and_remove_duplicates(folder_path, output_folder):
    """
    Remove exact duplicates and rename all image files sequentially.
    Args:
        folder_path: Path to the folder containing the images.
        output_folder: Path to the folder where renamed images will be saved.
    """
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)
    seen_hashes = set()
    count = 1

    for filename in sorted(os.listdir(folder_path)):  # Sort files to maintain order
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Filter for image files
            old_path = os.path.join(folder_path, filename)

            
            img_hash = calculate_hash(old_path)
            if img_hash in seen_hashes:
                print(f"Duplicate found: {filename} - Skipping")
                continue
            seen_hashes.add(img_hash)

            # Rename and save the image
            new_filename = f"{count}.jpg"  # Use `.jpg` or `.png` as per your preference
            new_path = os.path.join(output_folder, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
            count += 1

    print(f"Processing completed. Renamed {count - 1} unique files.")

# Example Usage
input_folder = "C:/Users/inoug/Desktop/Dataset/mountain"  # Replace with your input folder path
output_folder = "C:/Users/inoug/Desktop/Dataset/mountain_pose"  # Replace with your output folder path
rename_and_remove_duplicates(input_folder, output_folder)