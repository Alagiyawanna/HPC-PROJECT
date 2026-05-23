from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def png_to_pgm(png_path: Path, pgm_path: Path) -> None:
    # Convert a color or grayscale PNG into the PGM format expected by the C program.
    image = Image.open(png_path).convert("L")
    image.save(pgm_path)


def pgm_to_png(pgm_path: Path, png_path: Path) -> None:
    # Convert the generated PGM back to PNG so the result is easy to view in editors and browsers.
    image = Image.open(pgm_path).convert("L")
    image.save(png_path)


def open_image(path: Path) -> None:
    # Use the platform default image viewer when the user wants to open the result automatically.
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert input PNG to PGM, run the serial convolution, and convert output PGM to PNG."
    )
    # Keep the workflow flexible: callers can override filenames without touching the script.
    parser.add_argument("--input-png", default="image.png", help="Source PNG image")
    parser.add_argument("--input-pgm", default="image.pgm", help="Generated PGM input path")
    parser.add_argument("--output-pgm", default="output.pgm", help="Serial output PGM path")
    parser.add_argument("--output-png", default="output.png", help="Visualized output PNG path")
    parser.add_argument("--exe", default="serial_conv.exe", help="Serial executable path")
    parser.add_argument("--open-output", action="store_true", default=True, help="Open output.png after conversion")
    parser.add_argument("--no-open-output", dest="open_output", action="store_false", help="Do not open output.png")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    input_png = workdir / args.input_png
    input_pgm = workdir / args.input_pgm
    output_pgm = workdir / args.output_pgm
    output_png = workdir / args.output_png
    exe_path = workdir / args.exe

    # Reuse an existing PGM when available so repeated runs skip the PNG-to-PGM conversion step.
    if input_pgm.exists():
        print(f"Using existing input image: {input_pgm.name}")
    elif input_png.exists():
        png_to_pgm(input_png, input_pgm)
        print(f"Converted {input_png.name} -> {input_pgm.name}")
    else:
        print(f"Missing input image: expected {input_png.name} or {input_pgm.name}", file=sys.stderr)
        return 1

    # The serial executable now takes only input and output paths; blur is hardcoded in C.
    subprocess.run([str(exe_path), str(input_pgm), str(output_pgm)], check=True, cwd=workdir)

    # Convert the result for convenient viewing and optionally open it.
    pgm_to_png(output_pgm, output_png)
    print(f"Converted {output_pgm.name} -> {output_png.name}")
    if args.open_output:
        open_image(output_png)
        print(f"Opened {output_png.name}")
    print(f"Done: {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
