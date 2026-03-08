# DOCUMENT_METADATA

* File name: Unknown (Text Source)
* Page count: 1
* Detected language(s): English
* PDF metadata: Not available

# PAGE_1

## PAGE_LAYOUT

* Page dimensions: Undetectable from text input
* Orientation: Undetectable from text input
* Margins: Undetectable from text input
* Column structure: Single column (inferred)
* Header content: None detected
* Footer content: None detected
* Page numbers: None detected

## TEXT_BLOCKS

* Block ID: 001
* Position: Relative top of page
* Font family: Unknown
* Font size: Unknown
* Font weight/style: Unknown
* Text alignment: Unknown (assumed Left)
* Exact text content:
Project Proposal
EC7207 High Performance Computing


* Block ID: 002
* Position: Below Block 001
* Font family: Unknown
* Font size: Unknown
* Font weight/style: Unknown
* Text alignment: Unknown (assumed Left)
* Exact text content:

1. Project Description
This project addresses the computational challenges of image processing in modern applications
through parallel computing techniques. The key aspects include:
Image processing is widely used in computer vision, medical imaging, and autonomous
systems
2D convolution applies a filter (kernel) to an image for tasks like blurring, edge detection,
and sharpening
High-resolution images make serial execution computationally expensive
This project implements parallel image convolution using shared memory, distributed
memory, and hybrid approaches
Performance of parallel implementations will be compared against a serial baseline



* Block ID: 003
* Position: Below Block 002
* Font family: Unknown
* Font size: Unknown
* Font weight/style: Unknown
* Text alignment: Unknown (assumed Left)
* Exact text content:

2. Goal and Objectives
Goal
Design, implement, and evaluate high-performance parallel solutions for image
convolution
Objectives
Implement a serial version of image convolution as a baseline
Develop a shared memory parallel version using OpenMP
Develop a distributed memory parallel version using MPI
Develop a hybrid version combining $CPU+GPU(CUDA)$
Evaluate execution time, speedup, and efficiency for different thread/process counts
Verify accuracy using RMSE against the serial implementation



* Block ID: 004
* Position: Below Block 003
* Font family: Unknown
* Font size: Unknown
* Font weight/style: Unknown
* Text alignment: Unknown (assumed Left)
* Exact text content:

3. Methodology
Serial Implementation
Read input image
Apply convolution kernel to each pixel
Write output image
Shared Memory Parallelism (OpenMP)
Split image rows among multiple threads
Each thread computes convolution independently
Combine results to form output image
Distributed Memory Parallelism (MPI)
Split image into blocks (row-wise) among MPI processes
Exchange boundary (halo) rows between neighboring processes
Each process computes convolution for its block
Hybrid Parallelism (CPU + GPU)
CPU prepares image data and allocates memory
Image data is transferred from CPU to GPU memory
CUDA kernel executes convolution in parallel (one thread per pixel)
GPU performs massive parallel computation using CUDA cores
Results are transferred back to CPU memory
CPU collects final processed image and measures execution time
Performance Analysis
. Measure execution time for different image sizes, threads, and processes
Calculate speedup and efficiency
Compare results with serial output for accuracy (RMSE)



* Block ID: 005
* Position: Below Block 004
* Font family: Unknown
* Font size: Unknown
* Font weight/style: Unknown
* Text alignment: Unknown (assumed Left)
* Exact text content:

4. Expected Outcomes
Efficient serial, shared memory, distributed memory, and hybrid implementations
Demonstrated reduced execution time and improved scalability
Graphical analysis: speedup vs threads/processes
Verified accuracy: minimal error compared to serial implementation



## TABLES

None detected.

## IMAGES

None detected.

## EQUATIONS

* Block ID: EQ_001
* Position: Within TEXT_BLOCK_003
* LaTeX representation: $CPU+GPU(CUDA)$

* Plain-text representation: CPU+GPU(CUDA)



## FOOTNOTES

None detected.

## HYPERLINKS

None detected.

## SPECIAL ELEMENTS

None detected.

---

## FINAL VALIDATION STEP

1. Total number of pages processed: 1
2. Total number of images extracted: 0
3. Total number of tables extracted: 0
4. Confirmation that no content was omitted: Confirmed.
5. Confirmation that no content was added: Confirmed.
6. Confirmation that wording matches the PDF exactly: Confirmed.