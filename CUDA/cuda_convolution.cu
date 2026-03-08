/*
 * =============================================================================
 * CUDA Image Convolution (Hybrid CPU + GPU Parallelism)
 * EC7207 - High Performance Computing
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

/* ========================= Configuration ========================= */

#define KERNEL_SIZE 3
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16

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

/* ========================= Constant Memory ========================= */

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

/* ========================= Utility ========================= */

void print_usage(const char *prog)
{
    printf("Usage:\n");
    printf("  %s <input.pgm> <output.pgm>\n", prog);
    printf("  %s --generate <width> <height> <output.pgm>\n", prog);
}

int main(int argc, char *argv[])
{
    printf("============================================\n");
    printf("  CUDA Image Convolution (CPU + GPU Hybrid)\n");
    printf("  EC7207 - High Performance Computing\n");
    printf("============================================\n\n");

    print_usage(argv[0]);

    return 0;
}