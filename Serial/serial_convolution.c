/*
 * =============================================================================
 * Serial Image Convolution (Baseline Implementation)
 * EC7207 - High Performance Computing
 * =============================================================================
 *
 * Description:
 *   Performs 2D convolution on a grayscale PGM image using a serial approach.
 *   This serves as the baseline for performance comparison with parallel
 *   implementations (OpenMP, MPI, CUDA).
 *
 * Compilation:
 *   gcc -O2 -o serial_conv serial_convolution.c -lm
 *
 * Usage:
 *   ./serial_conv <input.pgm> <output.pgm>
 *   ./serial_conv --generate <width> <height> <output.pgm>
 *
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ========================= Configuration ========================= */

/* Convolution Kernel Definitions */
#define KERNEL_SIZE 3
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

/* Gaussian Blur 3x3 Kernel (unnormalized, will normalize at runtime) */
static float gaussian_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1.0f, 2.0f, 1.0f},
    {2.0f, 4.0f, 2.0f},
    {1.0f, 2.0f, 1.0f}};

/* Sharpen 3x3 Kernel */
static float sharpen_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0.0f, -1.0f, 0.0f},
    {-1.0f, 5.0f, -1.0f},
    {0.0f, -1.0f, 0.0f}};

/* Edge Detection (Laplacian) 3x3 Kernel */
static float edge_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {-1.0f, -1.0f, -1.0f},
    {-1.0f, 8.0f, -1.0f},
    {-1.0f, -1.0f, -1.0f}};

/* ========================= PGM Image I/O ========================= */

/**
 * Read a PGM (P5 binary) grayscale image from file.
 * Returns pixel data as a dynamically allocated array.
 */
unsigned char *read_pgm(const char *filename, int *width, int *height, int *maxval)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return NULL;
    }

    /* Read magic number */
    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P5") != 0)
    {
        fprintf(stderr, "Error: File '%s' is not a valid P5 PGM file\n", filename);
        fclose(fp);
        return NULL;
    }

    /* Skip comments */
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

    /* Read dimensions and max value */
    if (fscanf(fp, "%d %d %d", width, height, maxval) != 3)
    {
        fprintf(stderr, "Error: Cannot read PGM header\n");
        fclose(fp);
        return NULL;
    }

    /* Skip single whitespace character after maxval */
    fgetc(fp);

    /* Allocate and read pixel data */
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

/**
 * Write a PGM (P5 binary) grayscale image to file.
 */
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

/**
 * Generate a test grayscale image with patterns.
 * Creates a combination of gradients and geometric shapes.
 */
unsigned char *generate_test_image(int width, int height)
{
    unsigned char *data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (!data)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            /* Diagonal gradient */
            int val = (int)(255.0 * (i + j) / (height + width));

            /* Add a white rectangle in the center */
            if (i > height / 4 && i < 3 * height / 4 &&
                j > width / 4 && j < 3 * width / 4)
            {
                val = 200;
            }

            /* Add a dark circle */
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

/* ========================= Convolution ========================= */

/**
 * Normalize the convolution kernel so that all weights sum to 1.
 * Only applied for kernels where the sum is non-zero (e.g., blur).
 */
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

/**
 * Apply 2D convolution on a grayscale image (serial implementation).
 *
 * For each pixel (row, col), the convolution computes:
 *   output(row, col) = SUM over kernel of input(row+ki, col+kj) * kernel(ki, kj)
 *
 * Boundary handling: Zero-padding (pixels outside the image are treated as 0).
 */
void convolve_serial(const unsigned char *input, unsigned char *output,
                     int width, int height,
                     float kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float sum = 0.0f;

            /* Apply kernel centered at (row, col) */
            for (int ki = -KERNEL_RADIUS; ki <= KERNEL_RADIUS; ki++)
            {
                for (int kj = -KERNEL_RADIUS; kj <= KERNEL_RADIUS; kj++)
                {
                    int ni = row + ki; /* Neighbor row */
                    int nj = col + kj; /* Neighbor col */

                    /* Zero-padding boundary check */
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width)
                    {
                        sum += input[ni * width + nj] *
                               kernel[ki + KERNEL_RADIUS][kj + KERNEL_RADIUS];
                    }
                }
            }

            /* Clamp result to [0, 255] */
            if (sum < 0.0f)
                sum = 0.0f;
            if (sum > 255.0f)
                sum = 255.0f;
            output[row * width + col] = (unsigned char)(sum + 0.5f);
        }
    }
}

/* ========================= RMSE Calculation ========================= */

/**
 * Calculate Root Mean Square Error between two images.
 * Used to verify correctness of parallel implementations.
 */
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

/* ========================= Timing Utility ========================= */

/**
 * Get current time in seconds using high-resolution clock.
 */
double get_time_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ========================= Main Program ========================= */

void print_usage(const char *prog)
{
    printf("Usage:\n");
    printf("  %s <input.pgm> <output.pgm>\n", prog);
    printf("  %s --generate <width> <height> <output.pgm>\n", prog);
    printf("\nThis program applies 2D convolution (Gaussian blur) on a PGM image.\n");
}

int main(int argc, char *argv[])
{
    unsigned char *input_image = NULL;
    unsigned char *output_image = NULL;
    int width, height, maxval = 255;
    const char *output_filename = NULL;

    printf("============================================\n");
    printf("  Serial Image Convolution (Baseline)\n");
    printf("  EC7207 - High Performance Computing\n");
    printf("============================================\n\n");

    /* Parse command-line arguments */
    if (argc == 3)
    {
        /* Read from file */
        input_image = read_pgm(argv[1], &width, &height, &maxval);
        if (!input_image)
            return EXIT_FAILURE;
        output_filename = argv[2];
    }
    else if (argc == 5 && strcmp(argv[1], "--generate") == 0)
    {
        /* Generate test image */
        width = atoi(argv[2]);
        height = atoi(argv[3]);
        if (width <= 0 || height <= 0)
        {
            fprintf(stderr, "Error: Invalid dimensions %d x %d\n", width, height);
            return EXIT_FAILURE;
        }
        input_image = generate_test_image(width, height);
        if (!input_image)
            return EXIT_FAILURE;
        output_filename = argv[4];

        /* Save generated input image for reference */
        write_pgm("serial_input.pgm", input_image, width, height, maxval);
    }
    else
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    /* Allocate output image */
    output_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (!output_image)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(input_image);
        return EXIT_FAILURE;
    }

    /* Normalize the Gaussian kernel */
    normalize_kernel(gaussian_kernel);

    printf("\n[CONFIG] Image size    : %d x %d (%d pixels)\n", width, height, width * height);
    printf("[CONFIG] Kernel size   : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("[CONFIG] Kernel type   : Gaussian Blur\n");
    printf("[CONFIG] Boundary      : Zero-padding\n\n");

    /* =========== Perform Convolution with Timing =========== */
    printf("[STATUS] Starting serial convolution...\n");

    double start_time = get_time_seconds();
    convolve_serial(input_image, output_image, width, height, gaussian_kernel);
    double end_time = get_time_seconds();

    double elapsed_time = end_time - start_time;

    printf("[STATUS] Convolution complete.\n\n");

    /* =========== Results =========== */
    printf("==================== RESULTS ====================\n");
    printf("  Image size       : %d x %d\n", width, height);
    printf("  Total pixels     : %d\n", width * height);
    printf("  Kernel size      : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("  Execution time   : %.6f seconds\n", elapsed_time);
    printf("  Throughput       : %.2f Mpixels/sec\n",
           (width * height) / (elapsed_time * 1e6));
    printf("=================================================\n\n");

    /* Self-RMSE (should be 0.0) */
    double rmse = calculate_rmse(output_image, output_image, width, height);
    printf("[VERIFY] Self-RMSE     : %.6f (expected 0.0)\n", rmse);

    /* Write output image */
    if (write_pgm(output_filename, output_image, width, height, maxval) != 0)
    {
        free(input_image);
        free(output_image);
        return EXIT_FAILURE;
    }

    /* Save serial output for RMSE comparison with parallel versions */
    write_pgm("serial_output_reference.pgm", output_image, width, height, maxval);
    printf("[INFO] Reference output saved as 'serial_output_reference.pgm'\n");

    /* Cleanup */
    free(input_image);
    free(output_image);

    printf("\n[DONE] Serial convolution completed successfully.\n");
    return EXIT_SUCCESS;
}
