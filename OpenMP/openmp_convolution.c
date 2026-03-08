/*
 * =============================================================================
 * OpenMP Image Convolution (Shared Memory Parallelism)
 * EC7207 - High Performance Computing
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* ========================= Configuration ========================= */

#define KERNEL_SIZE 3
#define KERNEL_RADIUS (KERNEL_SIZE / 2)
#define DEFAULT_NUM_THREADS 4

/* ========================= Main Program ========================= */

void print_usage(const char *prog)
{
    printf("Usage:\n");
    printf("  %s <input.pgm> <output.pgm> [num_threads]\n", prog);
    printf("  %s --generate <width> <height> <output.pgm> [num_threads]\n", prog);
    printf("\nDefault threads: %d\n", DEFAULT_NUM_THREADS);
}

int main(int argc, char *argv[])
{
    printf("============================================\n");
    printf("  OpenMP Image Convolution (Shared Memory)\n");
    printf("  EC7207 - High Performance Computing\n");
    printf("============================================\n\n");

    print_usage(argv[0]);
    return 0;
}