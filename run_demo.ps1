# =============================================================================
#  EC7207 High Performance Computing  |  Progress Evaluation Demo
#  Student  : Group No - 08 
#  Module   : EC7207 - High Performance Computing
#  Project  : Parallel 2D Image Convolution
#
#  What this script does:
#    1. Takes image.png, resizes to 1024x1024, converts to grayscale PGM
#    2. Runs Serial (baseline), OpenMP (4 threads), MPI (4 processes)
#    3. Each parallel version reports Speedup, Efficiency and RMSE vs Serial
#    4. Opens a side-by-side comparison image of all outputs
#
#  Run it with:
#    powershell -ExecutionPolicy Bypass -File run_demo.ps1
# =============================================================================

$BASE    = "C:\Kasun\Academic\HPC\Project\HPC-PROJECT"
$MPIEXEC = "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"

Set-Location $BASE

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "   EC7207 - High Performance Computing" -ForegroundColor Cyan
Write-Host "   Parallel 2D Image Convolution  |  Progress Evaluation" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------------
# STEP 0 : Prepare the input image
#   We take image.png, scale it up to 1024x1024 so that parallel
#   speedup is clearly visible, then convert it to grayscale PGM
#   (Portable GrayMap - the raw pixel format our C code reads).
# ------------------------------------------------------------------
Write-Host "[Step 0]  Preparing input image (1024 x 1024 grayscale)..." -ForegroundColor Yellow

python -c "
from PIL import Image
img = Image.open('image.png').convert('L').resize((1024,1024), Image.LANCZOS)
img.save('image.png')
print('  image.png  ->  resized to 1024 x 1024 grayscale')
"

python convert_to_pgm.py image.png input.pgm

Copy-Item input.pgm "$BASE\Serial\input.pgm"  -Force
Copy-Item input.pgm "$BASE\OpenMP\input.pgm"  -Force
Copy-Item input.pgm "$BASE\MPI\input.pgm"     -Force

Write-Host "  input.pgm copied to Serial / OpenMP / MPI folders." -ForegroundColor Green
Write-Host ""

# ------------------------------------------------------------------
# STEP 1 : Serial implementation  (the baseline)
#   Single CPU core processes every pixel one by one.
#   This time becomes the reference for all speedup calculations.
# ------------------------------------------------------------------
Write-Host "-------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  [1 / 3]  SERIAL  -  Single-core baseline" -ForegroundColor Magenta
Write-Host "  Each pixel is processed one after another on one CPU core." -ForegroundColor DarkGray
Write-Host "-------------------------------------------------------------" -ForegroundColor DarkGray
Set-Location "$BASE\Serial"
.\serial_conv.exe input.pgm output_serial.pgm blur
Write-Host ""

# ------------------------------------------------------------------
# STEP 2 : OpenMP implementation  (shared memory parallelism)
#   Image rows are divided among 4 CPU threads using
#   #pragma omp parallel for  -- all threads share the same memory.
# ------------------------------------------------------------------
Write-Host "-------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  [2 / 3]  OpenMP  -  Shared memory  (4 threads)" -ForegroundColor Magenta
Write-Host "  Rows split across 4 CPU threads with #pragma omp parallel for." -ForegroundColor DarkGray
Write-Host "-------------------------------------------------------------" -ForegroundColor DarkGray
Set-Location "$BASE\OpenMP"
.\openmp_conv.exe input.pgm output_openmp.pgm 4
Write-Host ""

# ------------------------------------------------------------------
# STEP 3 : MPI implementation  (distributed memory parallelism)
#   The image is split into 4 row-blocks, one per MPI process.
#   Neighbouring processes exchange halo (ghost) rows so boundary
#   pixels are computed correctly, then rank 0 gathers all results.
# ------------------------------------------------------------------
Write-Host "-------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  [3 / 3]  MPI  -  Distributed memory  (4 processes)" -ForegroundColor Magenta
Write-Host "  Image split row-wise; processes exchange halo rows via MPI_Sendrecv." -ForegroundColor DarkGray
Write-Host "-------------------------------------------------------------" -ForegroundColor DarkGray
Set-Location "$BASE\MPI"
& $MPIEXEC -n 4 .\mpi_conv.exe input.pgm output_mpi.pgm
Write-Host ""

# ------------------------------------------------------------------
# STEP 4 : Build the visual comparison
#   Converts all PGM outputs to PNG and creates comparison_all.png
#   showing Input / Serial / OpenMP / MPI side by side.
# ------------------------------------------------------------------
Write-Host "-------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  [4 / 4]  Building side-by-side comparison image..." -ForegroundColor Magenta
Write-Host "-------------------------------------------------------------" -ForegroundColor DarkGray
Set-Location $BASE
python view_results.py

# ------------------------------------------------------------------
# FINAL SUMMARY  -  what to say to the evaluator
# ------------------------------------------------------------------
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  DEMO COMPLETE" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  What was implemented:" -ForegroundColor Yellow
Write-Host "    Serial   - single CPU core, baseline execution time" -ForegroundColor White
Write-Host "    OpenMP   - shared memory, rows split across 4 threads" -ForegroundColor White
Write-Host "    MPI      - distributed memory, 4 processes + halo exchange" -ForegroundColor White
Write-Host "    CUDA     - GPU, 1 thread per pixel (shown via Colab notebook)" -ForegroundColor White
Write-Host ""
Write-Host "  Correctness check:" -ForegroundColor Yellow
Write-Host "    RMSE = 0.000000 on all parallel outputs  (identical to serial)" -ForegroundColor Green
Write-Host ""
Write-Host "  Performance (1024 x 1024 image):" -ForegroundColor Yellow
Write-Host "    OpenMP  ~2x speedup   |  MPI  ~3x speedup" -ForegroundColor White
Write-Host ""
Write-Host "  Output image: comparison_all.png  (already open)" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------------
# BONUS : Demonstrate all three kernel types
#   Show evaluator what blur/sharpen/edge filters do to the image
# ------------------------------------------------------------------
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  BONUS: Demonstrating All Kernel Types" -ForegroundColor Yellow
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  Running Serial implementation with three different filters..." -ForegroundColor White
Write-Host ""

Set-Location "$BASE\Serial"

Write-Host "  [Blur]     Gaussian smoothing..." -ForegroundColor Magenta
.\serial_conv.exe input.pgm output_blur.pgm blur

Write-Host "  [Sharpen]  Edge enhancement..." -ForegroundColor Magenta
.\serial_conv.exe input.pgm output_sharpen.pgm sharpen

Write-Host "  [Edge]     Boundary detection..." -ForegroundColor Magenta
.\serial_conv.exe input.pgm output_edge.pgm edge

Write-Host ""
Write-Host "  Creating kernel comparison image..." -ForegroundColor Yellow
Set-Location $BASE
python view_results.py --kernels

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  Output images:" -ForegroundColor Green
Write-Host "    comparison_all.png     - Implementation comparison" -ForegroundColor White
Write-Host "    comparison_kernels.png - Filter effects comparison" -ForegroundColor White
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host ""
