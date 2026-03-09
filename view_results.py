"""
view_results.py
---------------
Converts all PGM output images to PNG and builds a labelled
side-by-side comparison image (comparison_all.png) ready for
the EC7207 HPC Project progress evaluation.

Usage:
    python view_results.py            # Compare implementations (Serial/OpenMP/MPI)
    python view_results.py --kernels  # Compare kernel types (Blur/Sharpen/Edge)
"""

from PIL import Image, ImageDraw, ImageFont
import os, subprocess, sys

BASE = os.path.dirname(os.path.abspath(__file__))

# Images to include in the comparison with their display labels
IMAGES = [
    ("Serial\\input.pgm",         "Input Image",        "(original grayscale)"),
    ("Serial\\output_serial.pgm", "Serial  [Baseline]", "single CPU core"),
    ("OpenMP\\output_openmp.pgm", "OpenMP  [4 Threads]","shared memory | RMSE = 0.000000"),
    ("MPI\\output_mpi.pgm",       "MPI  [4 Processes]", "distributed memory | RMSE = 0.000000"),
]

# Kernel comparison images (for --kernels mode)
KERNEL_IMAGES = [
    ("Serial\\input.pgm",         "Input Image",     "(original grayscale)"),
    ("Serial\\output_blur.pgm",   "Gaussian Blur",   "smooths and reduces noise"),
    ("Serial\\output_sharpen.pgm","Sharpen Filter",  "enhances edges and details"),
    ("Serial\\output_edge.pgm",   "Edge Detection",  "highlights boundaries (Laplacian)"),
]

THUMB   = 300   # pixel size of each thumbnail in the comparison
PAD     = 14
LABEL_H = 52    # height reserved for two lines of labels below each image

def load_image(path):
    return Image.open(path).convert('L')

def convert_to_png(pgm_path):
    png_path = pgm_path.replace('.pgm', '.png')
    img = load_image(pgm_path)
    img.save(png_path)
    print(f"  Converted  {os.path.basename(pgm_path):35s}  ->  {os.path.basename(png_path)}  ({img.width}x{img.height})")
    return png_path, img

def generate_comparison(images_list, output_name, title):
    """Generate a side-by-side comparison image from a list of images"""
    print(f"\n[2]  Building {output_name} ...")
    
    COLS = len(images_list)
    W    = COLS * (THUMB + PAD) + PAD
    H    = PAD + THUMB + LABEL_H + PAD

    # Light grey background
    comparison = Image.new('RGB', (W, H), color=(230, 230, 230))
    draw = ImageDraw.Draw(comparison)

    # Try to load a readable font; fall back to default if not available
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 15)
        font_sub   = ImageFont.truetype("arial.ttf",   12)
    except:
        font_title = ImageFont.load_default()
        font_sub   = font_title

    for i, (png_path, img_title, subtitle, img) in enumerate(images_list):
        # Paste thumbnail
        thumb = img.resize((THUMB, THUMB), Image.LANCZOS).convert('RGB')
        x = PAD + i * (THUMB + PAD)
        y = PAD
        comparison.paste(thumb, (x, y))

        # Thin border around each thumbnail
        draw.rectangle([x-1, y-1, x+THUMB, y+THUMB], outline=(80,80,80), width=1)

        # Title (bold) and subtitle below image
        draw.text((x + 4, y + THUMB + 6),  img_title, fill=(20, 20, 20),   font=font_title)
        draw.text((x + 4, y + THUMB + 26), subtitle,  fill=(60, 60, 60),   font=font_sub)

    comp_path = os.path.join(BASE, output_name)
    comparison.save(comp_path)
    print(f"  Saved  ->  {comp_path}")
    return comp_path

# ===== Main Execution =====
if __name__ == '__main__':
    # Check if we're in kernel comparison mode
    kernel_mode = len(sys.argv) > 1 and sys.argv[1] == '--kernels'
    
    if kernel_mode:
        images_to_process = KERNEL_IMAGES
        output_filename = "comparison_kernels.png"
        title_text = "Kernel Comparison"
    else:
        images_to_process = IMAGES
        output_filename = "comparison_all.png"
        title_text = "Implementation Comparison"
    
    print("=" * 60)
    print(f"  EC7207 HPC  |  {title_text}  |  Results")
    print("=" * 60)

    # ── 1. Convert each PGM output to PNG ───────────────────────────
    print("\n[1]  Converting PGM outputs to PNG ...")
    converted = []
    for rel_path, title, subtitle in images_to_process:
        pgm = os.path.join(BASE, rel_path)
        if os.path.exists(pgm):
            png_path, img = convert_to_png(pgm)
            converted.append((png_path, title, subtitle, img))
        else:
            print(f"  WARNING  {pgm}  not found - skipping.")

    # ── 2. Build the side-by-side comparison image ──────────────────
    comp_path = generate_comparison(converted, output_filename, title_text)

    # ── 3. Open only the comparison image (cleaner for evaluation) ──
    print("\n[3]  Opening comparison image ...")
    subprocess.Popen(['explorer.exe', comp_path])

    print("\n" + "=" * 60)
    if kernel_mode:
        print("  Kernel comparison complete.")
        print(f"  Show  {output_filename}  to demonstrate different filter effects.")
    else:
        print("  Implementation comparison complete.")
        print(f"  Show  {output_filename}  to the evaluator.")
    print("=" * 60)
