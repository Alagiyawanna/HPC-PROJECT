/*
 * =============================================================================
 * CUDA Image Convolution (Hybrid CPU + GPU Parallelism)
 * EC7207 - High Performance Computing
 * =============================================================================
 *
 * Description:
 *   Performs 2D convolution on a grayscale PGM image using CUDA.
 *   The image data is transferred from CPU to GPU, a CUDA kernel executes
 *   the convolution in parallel (one thread per pixel), and the result is
 *   transferred back to the CPU. Uses GPU shared memory tiling for performance.
 *   Results are compared against a serial CPU baseline using RMSE.
 *
 * Compilation:
 *   nvcc -O2 -o cuda_conv cuda_convolution.cu -lm
 *
 * Usage:
 *   ./cuda_conv <input.pgm> <output.pgm>
 *   ./cuda_conv --generate <width> <height> <output.pgm>
 *
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/* ========================= Configuration ========================= */

#define KERNEL_SIZE 3
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

/* CUDA block dimensions */
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

/* Shared memory tile dimensions (block + halo on each side) */
#define TILE_WIDTH (BLOCK_WIDTH + 2 * KERNEL_RADIUS)
#define TILE_HEIGHT (BLOCK_HEIGHT + 2 * KERNEL_RADIUS)

/* ========================= CUDA Error Checking ========================= */

#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/* ========================= Constant Memory for Kernel ========================= */

/* Store convolution kernel in GPU constant memory for fast cached access */
__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

/* ========================= CUDA Kernels ========================= */

/**
 * Naive CUDA convolution kernel.
 * Each thread computes one output pixel.
 * Reads directly from global memory.
 */
__global__ void convolve_naive_kernel(const unsigned char *input,
                                      unsigned char *output,
                                      int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width)
        return;

    float sum = 0.0f;

    for (int ki = -KERNEL_RADIUS; ki <= KERNEL_RADIUS; ki++)
    {
        for (int kj = -KERNEL_RADIUS; kj <= KERNEL_RADIUS; kj++)
        {
            int ni = row + ki;
            int nj = col + kj;

            if (ni >= 0 && ni < height && nj >= 0 && nj < width)
            {
                sum += (float)input[ni * width + nj] *
                       d_kernel[(ki + KERNEL_RADIUS) * KERNEL_SIZE + (kj + KERNEL_RADIUS)];
            }
        }
    }

    if (sum < 0.0f)
        sum = 0.0f;
    if (sum > 255.0f)
        sum = 255.0f;
    output[row * width + col] = (unsigned char)(sum + 0.5f);
}

/**
 * Tiled CUDA convolution kernel using shared memory.
 * Each block loads a tile (including halo) into shared memory,
 * then computes convolution using the fast shared memory.
 * This reduces global memory accesses significantly.
 */
__global__ void convolve_tiled_kernel(const unsigned char *input,
                                      unsigned char *output,
                                      int width, int height)
{
    /* Shared memory tile for this block (includes halo pixels) */
    __shared__ float tile[TILE_HEIGHT][TILE_WIDTH];

    /* Output pixel coordinates */
    int col = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    int row = blockIdx.y * BLOCK_HEIGHT + threadIdx.y;

    /* Input coordinates (shifted by halo radius) */
    int input_row = row - KERNEL_RADIUS;
    int input_col = col - KERNEL_RADIUS;

    /* Load tile into shared memory */
    /* Each thread loads one element of the tile */
    /* We need TILE_WIDTH * TILE_HEIGHT elements but have BLOCK_WIDTH * BLOCK_HEIGHT threads */
    /* So some threads load multiple elements */

    for (int ty = threadIdx.y; ty < TILE_HEIGHT; ty += BLOCK_HEIGHT)
    {
        for (int tx = threadIdx.x; tx < TILE_WIDTH; tx += BLOCK_WIDTH)
        {
            int src_row = (int)(blockIdx.y * BLOCK_HEIGHT) - KERNEL_RADIUS + ty;
            int src_col = (int)(blockIdx.x * BLOCK_WIDTH) - KERNEL_RADIUS + tx;

            if (src_row >= 0 && src_row < height && src_col >= 0 && src_col < width)
            {
                tile[ty][tx] = (float)input[src_row * width + src_col];
            }
            else
            {
                tile[ty][tx] = 0.0f; /* Zero-padding */
            }
        }
    }

    __syncthreads();

    /* Compute convolution for this pixel */
    if (row < height && col < width)
    {
        float sum = 0.0f;

        for (int ki = 0; ki < KERNEL_SIZE; ki++)
        {
            for (int kj = 0; kj < KERNEL_SIZE; kj++)
            {
                sum += tile[threadIdx.y + ki][threadIdx.x + kj] *
                       d_kernel[ki * KERNEL_SIZE + kj];
            }
        }

        if (sum < 0.0f)
            sum = 0.0f;
        if (sum > 255.0f)
            sum = 255.0f;
        output[row * width + col] = (unsigned char)(sum + 0.5f);
    }
}

/* ========================= Host Functions ========================= */

/* PGM Image I/O (host side) */

unsigned char *read_pgm(const char *filename, int *width, int *height, int *maxval)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return NULL;
    }

    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P5") != 0)
    {
        fprintf(stderr, "Error: File '%s' is not a valid P5 PGM file\n", filename);
        fclose(fp);
        return NULL;
    }

    int ch;
    while ((ch = fgetc(fp)) == '#' || ch == '\n' || ch == '\r' || ch == ' ')
    {
        if (ch == '#')
        {
            while ((ch = fgetc(fp)) != '\n' && ch != EOF)
                ;
        }
    }
    ungetc(ch, fp);

    if (fscanf(fp, "%d %d %d", width, height, maxval) != 3)
    {
        fprintf(stderr, "Error: Cannot read PGM header\n");
        fclose(fp);
        return NULL;
    }
    fgetc(fp);

    int size = (*width) * (*height);
    unsigned char *data = (unsigned char *)malloc(size * sizeof(unsigned char));
    if (!data)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }

    if (fread(data, sizeof(unsigned char), size, fp) != (size_t)size)
    {
        fprintf(stderr, "Error: Cannot read pixel data\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    printf("[INFO] Read PGM image: %d x %d, maxval=%d\n", *width, *height, *maxval);
    return data;
}

int write_pgm(const char *filename, unsigned char *data, int width, int height, int maxval)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot create file '%s'\n", filename);
        return -1;
    }

    fprintf(fp, "P5\n%d %d\n%d\n", width, height, maxval);
    fwrite(data, sizeof(unsigned char), width * height, fp);
    fclose(fp);

    printf("[INFO] Wrote PGM image: %s (%d x %d)\n", filename, width, height);
    return 0;
}

unsigned char *generate_test_image(int width, int height)
{
    unsigned char *data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (!data)
        return NULL;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int val = (int)(255.0 * (i + j) / (height + width));
            if (i > height / 4 && i < 3 * height / 4 &&
                j > width / 4 && j < 3 * width / 4)
            {
                val = 200;
            }
            int ci = height / 2, cj = width / 2;
            int radius = height < width ? height / 6 : width / 6;
            if ((i - ci) * (i - ci) + (j - cj) * (j - cj) < radius * radius)
            {
                val = 50;
            }
            data[i * width + j] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
        }
    }

    printf("[INFO] Generated test image: %d x %d\n", width, height);
    return data;
}

/* Serial convolution (CPU baseline) */
void convolve_serial(const unsigned char *input, unsigned char *output,
                     int width, int height, float kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float sum = 0.0f;
            for (int ki = -KERNEL_RADIUS; ki <= KERNEL_RADIUS; ki++)
            {
                for (int kj = -KERNEL_RADIUS; kj <= KERNEL_RADIUS; kj++)
                {
                    int ni = row + ki;
                    int nj = col + kj;
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width)
                    {
                        sum += input[ni * width + nj] *
                               kernel[ki + KERNEL_RADIUS][kj + KERNEL_RADIUS];
                    }
                }
            }
            if (sum < 0.0f)
                sum = 0.0f;
            if (sum > 255.0f)
                sum = 255.0f;
            output[row * width + col] = (unsigned char)(sum + 0.5f);
        }
    }
}

/* Normalize kernel */
void normalize_kernel(float kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    float sum = 0.0f;
    for (int i = 0; i < KERNEL_SIZE; i++)
        for (int j = 0; j < KERNEL_SIZE; j++)
            sum += kernel[i][j];

    if (fabs(sum) > 1e-6)
    {
        for (int i = 0; i < KERNEL_SIZE; i++)
            for (int j = 0; j < KERNEL_SIZE; j++)
                kernel[i][j] /= sum;
    }
}

/* RMSE calculation */
double calculate_rmse(const unsigned char *img1, const unsigned char *img2,
                      int width, int height)
{
    double sum_sq = 0.0;
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++)
    {
        double diff = (double)img1[i] - (double)img2[i];
        sum_sq += diff * diff;
    }

    return sqrt(sum_sq / total_pixels);
}

/* Print GPU device info */
void print_gpu_info()
{
    int device;
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("[GPU] Device         : %s\n", prop.name);
    printf("[GPU] Compute Cap    : %d.%d\n", prop.major, prop.minor);
    printf("[GPU] SM Count       : %d\n", prop.multiProcessorCount);
    printf("[GPU] Global Memory  : %.1f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("[GPU] Shared Mem/Blk : %zu bytes\n", prop.sharedMemPerBlock);
    printf("[GPU] Max Threads/Blk: %d\n", prop.maxThreadsPerBlock);
    printf("[GPU] Warp Size      : %d\n", prop.warpSize);
    printf("\n");
}

/* ========================= Main Program ========================= */

void print_usage(const char *prog)
{
    printf("Usage:\n");
    printf("  %s <input.pgm> <output.pgm>\n", prog);
    printf("  %s --generate <width> <height> <output.pgm>\n", prog);
}

int main(int argc, char *argv[])
{
    unsigned char *h_input = NULL;         /* Host input image */
    unsigned char *h_output_gpu = NULL;    /* Host output from GPU */
    unsigned char *h_output_serial = NULL; /* Host output from CPU serial */
    unsigned char *d_input = NULL;         /* Device input image */
    unsigned char *d_output = NULL;        /* Device output image */
    int width, height, maxval = 255;
    const char *output_filename = NULL;

    /* Gaussian Blur 3x3 Kernel */
    float h_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}};

    printf("============================================\n");
    printf("  CUDA Image Convolution (CPU + GPU Hybrid)\n");
    printf("  EC7207 - High Performance Computing\n");
    printf("============================================\n\n");

    /* Print GPU info */
    print_gpu_info();

    /* Parse command-line arguments */
    if (argc >= 3 && strcmp(argv[1], "--generate") != 0)
    {
        h_input = read_pgm(argv[1], &width, &height, &maxval);
        if (!h_input)
            return EXIT_FAILURE;
        output_filename = argv[2];
    }
    else if (argc >= 5 && strcmp(argv[1], "--generate") == 0)
    {
        width = atoi(argv[2]);
        height = atoi(argv[3]);
        if (width <= 0 || height <= 0)
        {
            fprintf(stderr, "Error: Invalid dimensions %d x %d\n", width, height);
            return EXIT_FAILURE;
        }
        h_input = generate_test_image(width, height);
        if (!h_input)
            return EXIT_FAILURE;
        output_filename = argv[4];
    }
    else
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int img_size = width * height;

    /* Allocate host buffers */
    h_output_gpu = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    h_output_serial = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    if (!h_output_gpu || !h_output_serial)
    {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    /* Normalize kernel */
    normalize_kernel(h_kernel);

    printf("[CONFIG] Image size    : %d x %d (%d pixels)\n", width, height, img_size);
    printf("[CONFIG] Kernel size   : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("[CONFIG] Kernel type   : Gaussian Blur\n");
    printf("[CONFIG] Block size    : %d x %d\n", BLOCK_WIDTH, BLOCK_HEIGHT);
    printf("[CONFIG] Boundary      : Zero-padding\n\n");

    /* =========== Copy Kernel to Constant Memory =========== */
    float flat_kernel[KERNEL_SIZE * KERNEL_SIZE];
    for (int i = 0; i < KERNEL_SIZE; i++)
        for (int j = 0; j < KERNEL_SIZE; j++)
            flat_kernel[i * KERNEL_SIZE + j] = h_kernel[i][j];

    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, flat_kernel,
                                  KERNEL_SIZE * KERNEL_SIZE * sizeof(float)));

    /* =========== Allocate Device Memory =========== */
    CUDA_CHECK(cudaMalloc((void **)&d_input, img_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, img_size * sizeof(unsigned char)));

    /* =========== CUDA Events for Timing =========== */
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;

    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&stop_d2h));

    /* =========== GPU Execution =========== */
    printf("[STATUS] Starting CUDA convolution...\n");

    /* Grid and block dimensions */
    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridDim((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
                 (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

    printf("[LAUNCH] Grid  : %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("[LAUNCH] Block : %d x %d threads\n", blockDim.x, blockDim.y);
    printf("[LAUNCH] Total threads: %d\n\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);

    /* --- Total GPU timing start --- */
    CUDA_CHECK(cudaEventRecord(start_total));

    /* Step 1: Host to Device transfer */
    CUDA_CHECK(cudaEventRecord(start_h2d));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, img_size * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_h2d));

    /* Step 2: Launch tiled convolution kernel */
    CUDA_CHECK(cudaEventRecord(start_kernel));
    convolve_tiled_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop_kernel));

    /* Step 3: Device to Host transfer */
    CUDA_CHECK(cudaEventRecord(start_d2h));
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, img_size * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_d2h));

    /* --- Total GPU timing end --- */
    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    /* Get timing results (in milliseconds) */
    float time_h2d, time_kernel, time_d2h, time_total;
    CUDA_CHECK(cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h));
    CUDA_CHECK(cudaEventElapsedTime(&time_total, start_total, stop_total));

    printf("[STATUS] CUDA convolution complete.\n\n");

    /* =========== Serial Baseline =========== */
    printf("[STATUS] Running serial CPU baseline for comparison...\n");

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    convolve_serial(h_input, h_output_serial, width, height, h_kernel);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double serial_time = (ts_end.tv_sec - ts_start.tv_sec) +
                         (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;
    printf("[STATUS] Serial complete: %.6f seconds\n\n", serial_time);

    /* =========== Performance Analysis =========== */
    double gpu_total_sec = time_total / 1000.0;
    double gpu_kernel_sec = time_kernel / 1000.0;
    double speedup_total = serial_time / gpu_total_sec;
    double speedup_kernel = serial_time / gpu_kernel_sec;
    double rmse = calculate_rmse(h_output_serial, h_output_gpu, width, height);

    printf("==================== RESULTS ====================\n");
    printf("  Image size       : %d x %d\n", width, height);
    printf("  Total pixels     : %d\n", img_size);
    printf("  Kernel size      : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("  -----------------------------------------\n");
    printf("  Serial CPU time  : %.6f seconds\n", serial_time);
    printf("  -----------------------------------------\n");
    printf("  GPU H2D transfer : %.3f ms\n", time_h2d);
    printf("  GPU kernel exec  : %.3f ms\n", time_kernel);
    printf("  GPU D2H transfer : %.3f ms\n", time_d2h);
    printf("  GPU total time   : %.3f ms (%.6f sec)\n", time_total, gpu_total_sec);
    printf("  -----------------------------------------\n");
    printf("  Speedup (total)  : %.4f x (incl. transfers)\n", speedup_total);
    printf("  Speedup (kernel) : %.4f x (kernel only)\n", speedup_kernel);
    printf("  Throughput       : %.2f Mpixels/sec (total)\n",
           img_size / (gpu_total_sec * 1e6));
    printf("  Throughput       : %.2f Mpixels/sec (kernel)\n",
           img_size / (gpu_kernel_sec * 1e6));
    printf("  RMSE             : %.6f\n", rmse);
    printf("=================================================\n\n");

    if (rmse < 1e-6)
    {
        printf("[VERIFY] PASSED - Output matches serial baseline exactly.\n");
    }
    else if (rmse < 1.0)
    {
        printf("[VERIFY] PASSED - Output within acceptable tolerance (RMSE < 1.0).\n");
    }
    else
    {
        printf("[VERIFY] WARNING - Significant difference from serial baseline!\n");
    }

    /* Write output image */
    write_pgm(output_filename, h_output_gpu, width, height, maxval);

    /* =========== Cleanup =========== */
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_h2d));
    CUDA_CHECK(cudaEventDestroy(stop_h2d));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_d2h));
    CUDA_CHECK(cudaEventDestroy(stop_d2h));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    free(h_input);
    free(h_output_gpu);
    free(h_output_serial);

    printf("\n[DONE] CUDA convolution completed successfully.\n");
    return EXIT_SUCCESS;
}
