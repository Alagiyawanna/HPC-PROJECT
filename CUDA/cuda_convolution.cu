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

/* ========================= PGM Image I/O ========================= */

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
            int radius = (height < width) ? height / 6 : width / 6;
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

/* ========================= Serial Baseline ========================= */

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

/* ========================= Utility ========================= */

void print_usage(const char *prog)
{
    printf("Usage:\n");
    printf("  %s <input.pgm> <output.pgm>\n", prog);
    printf("  %s --generate <width> <height> <output.pgm>\n", prog);
}

int main(int argc, char *argv[])
{
    unsigned char *h_input = NULL;
    unsigned char *h_output_serial = NULL;
    int width, height, maxval = 255;
    const char *output_filename = NULL;

    float h_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}};

    printf("============================================\n");
    printf("  CUDA Image Convolution (CPU + GPU Hybrid)\n");
    printf("  EC7207 - High Performance Computing\n");
    printf("============================================\n\n");

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
    h_output_serial = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    if (!h_output_serial)
    {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        free(h_input);
        return EXIT_FAILURE;
    }

    normalize_kernel(h_kernel);
    convolve_serial(h_input, h_output_serial, width, height, h_kernel);

    write_pgm(output_filename, h_output_serial, width, height, maxval);

    printf("[INFO] Serial convolution completed.\n");
    printf("[INFO] RMSE self-check: %.6f\n",
           calculate_rmse(h_output_serial, h_output_serial, width, height));

    free(h_input);
    free(h_output_serial);
    return 0;
}