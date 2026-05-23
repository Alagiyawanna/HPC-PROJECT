# Serial Folder Workflow

This folder now supports a simple end-to-end workflow:

1. Convert `image.png` to `image.pgm` if needed, or use an existing `image.pgm`
2. Run the serial convolution program to produce `output.pgm`
3. Convert `output.pgm` to `output.png` so you can view the result easily

## Build

Open a terminal in the `Serial` folder and compile the program:

```bat
gcc -O2 -o serial_conv.exe serial_convolution.c -lm
```

## Run

Make sure `image.png` is in this folder, then run:

```bat
python run_serial_workflow.py
```

This creates:

- `image.pgm` if `image.png` was provided
- `output.pgm`
- `output.png`

After conversion, `output.png` opens automatically in the default image viewer.

## Custom kernel

You can choose a kernel while running the workflow:

```bat
python run_serial_workflow.py --kernel sharpen
python run_serial_workflow.py --kernel edge
```

If you do not want the image viewer to open automatically, use:

```bat
python run_serial_workflow.py --no-open-output
```

## Notes

- `output.png` is just a visual copy of `output.pgm`.
- The workflow uses Pillow for PNG/PGM conversion.
- If Pillow is missing, install it with:

```bat
pip install pillow
```
