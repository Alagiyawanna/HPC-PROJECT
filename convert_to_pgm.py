"""
convert_to_pgm.py
-----------------
Converts any image (PNG, JPG, BMP, etc.) to grayscale PGM (P5 binary) format.
Used for EC7207 HPC Project - Image Convolution.

Usage:
    python convert_to_pgm.py <input_image> <output.pgm>

Example:
    python convert_to_pgm.py photo.png input.pgm
    python convert_to_pgm.py photo.jpg input.pgm
"""

import sys
from PIL import Image

def convert_to_pgm(input_path, output_path):
    img = Image.open(input_path).convert('L')   # convert to grayscale
    width, height = img.size
    pixels = img.tobytes()

    with open(output_path, 'wb') as f:
        # PGM P5 header
        header = f"P5\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(pixels)

    print(f"Converted '{input_path}' -> '{output_path}' ({width}x{height} grayscale PGM)")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_to_pgm.py <input_image> <output.pgm>")
        sys.exit(1)
    convert_to_pgm(sys.argv[1], sys.argv[2])
