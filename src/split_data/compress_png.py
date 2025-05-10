from PIL import Image
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor


def compress_png(input_path, output_path, max_size_mb=1.0):
    """
    Compress a PNG image using optimized settings.
    """
    try:
        # Open the image
        img = Image.open(input_path)

        # Convert to RGB if image is in RGBA mode
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Calculate target size in pixels based on original image
        original_size = os.path.getsize(input_path) / (1024 * 1024)  # Convert to MB
        scale_factor = 1.0

        if original_size > max_size_mb:
            scale_factor = (max_size_mb / original_size) ** 0.5
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save with optimized settings
        img.save(output_path, 'PNG', optimize=True, quality=85)

        final_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Compressed {input_path} to {final_size:.2f}MB")
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def process_single_file(args):
    """Helper function for parallel processing"""
    file_path, output_folder, max_size_mb = args
    output_path = output_folder / file_path.name
    return compress_png(str(file_path), str(output_path), max_size_mb)


def process_folder(folder_path, max_size_mb=1.0):
    """
    Process all PNG files in the given folder that don't start with 'CXR'.
    Uses parallel processing for better performance.
    """
    folder = Path(folder_path)
    output_folder = folder
    output_folder.mkdir(exist_ok=True)

    # Get list of files to process
    files_to_process = [
        f for f in folder.glob('*.png')
        if not f.name.startswith('CXR')
    ]

    # Prepare arguments for parallel processing
    args_list = [
        (file_path, output_folder, max_size_mb)
        for file_path in files_to_process
    ]

    # Process files in parallel
    successful = 0
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single_file, args_list))
        successful = sum(results)

    print(f"\nCompression complete!")
    print(f"Processed {len(files_to_process)} files")
    print(f"Successfully compressed: {successful}")
    print(f"Failed: {len(files_to_process) - successful}")
    print(f"\nCompressed files are saved in the 'compressed' subfolder")


if __name__ == "__main__":
    # Get folder path from command line or use current directory
    folder_path = "images"

    # Get max size from command line (default 1MB)
    max_size_mb = 0.16

    process_folder(folder_path, max_size_mb)