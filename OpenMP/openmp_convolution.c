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

static float gaussian_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1.0f, 2.0f, 1.0f},
    {2.0f, 4.0f, 2.0f},
    {1.0f, 2.0f, 1.0f}};

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
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

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

/* ========================= Kernel Utility ========================= */

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

/* ========================= Serial Convolution ========================= */

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
    unsigned char *input_image = NULL;
    unsigned char *output_serial = NULL;
    int width, height, maxval = 255;
    const char *output_filename = NULL;
    int num_threads = DEFAULT_NUM_THREADS;

    printf("============================================\n");
    printf("  OpenMP Image Convolution (Shared Memory)\n");
    printf("  EC7207 - High Performance Computing\n");
    printf("============================================\n\n");

    if (argc >= 3 && strcmp(argv[1], "--generate") != 0)
    {
        input_image = read_pgm(argv[1], &width, &height, &maxval);
        if (!input_image)
            return EXIT_FAILURE;
        output_filename = argv[2];
        if (argc >= 4)
            num_threads = atoi(argv[3]);
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

        input_image = generate_test_image(width, height);
        if (!input_image)
            return EXIT_FAILURE;

        output_filename = argv[4];
        if (argc >= 6)
            num_threads = atoi(argv[5]);
    }
    else
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int img_size = width * height;
    output_serial = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    if (!output_serial)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(input_image);
        free(output_serial);
        return EXIT_FAILURE;
    }

    normalize_kernel(gaussian_kernel);
    convolve_serial(input_image, output_serial, width, height, gaussian_kernel);
    write_pgm(output_filename, output_serial, width, height, maxval);

    printf("[INFO] Serial baseline completed.\n");

    free(input_image);
    free(output_serial);
    return 0;
}