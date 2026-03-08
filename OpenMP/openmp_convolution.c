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

/* ========================= Convolution Helpers ========================= */

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

void convolve_openmp(const unsigned char *input, unsigned char *output,
                     int width, int height, float kernel[KERNEL_SIZE][KERNEL_SIZE],
                     int num_threads)
{
    #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
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
    {
        for (int j = 0; j < KERNEL_SIZE; j++)
        {
            sum += kernel[i][j];
        }
    }

    if (fabs(sum) > 1e-6)
    {
        for (int i = 0; i < KERNEL_SIZE; i++)
        {
            for (int j = 0; j < KERNEL_SIZE; j++)
            {
                kernel[i][j] /= sum;
            }
        }
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
    unsigned char *output_openmp = NULL;
    unsigned char *output_serial = NULL;
    float kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}};
    printf("============================================\n");
    printf("  OpenMP Image Convolution (Shared Memory)\n");
    printf("  EC7207 - High Performance Computing\n");
    printf("============================================\n\n");

    int width, height, maxval = 255;
    const char *output_filename = NULL;
    int num_threads = DEFAULT_NUM_THREADS;

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

    if (num_threads <= 0)
    {
        fprintf(stderr, "Error: Number of threads must be positive\n");
        free(input_image);
        return EXIT_FAILURE;
    }

    int img_size = width * height;
    output_openmp = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    output_serial = (unsigned char *)malloc(img_size * sizeof(unsigned char));
    if (!output_openmp || !output_serial)
    {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(input_image);
        free(output_openmp);
        free(output_serial);
        return EXIT_FAILURE;
    }

    normalize_kernel(kernel);
    omp_set_num_threads(num_threads);

    printf("[CONFIG] Image size        : %d x %d (%d pixels)\n", width, height, img_size);
    printf("[CONFIG] Kernel size       : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("[CONFIG] Kernel type       : Gaussian Blur\n");
    printf("[CONFIG] Boundary          : Zero-padding\n");
    printf("[CONFIG] Output file       : %s\n", output_filename);
    printf("[CONFIG] Threads requested : %d\n", num_threads);
    printf("[CONFIG] Max OpenMP threads: %d\n\n", omp_get_max_threads());

    printf("[STATUS] Running serial CPU baseline...\n");
    double serial_start = omp_get_wtime();
    convolve_serial(input_image, output_serial, width, height, kernel);
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;
    printf("[STATUS] Serial complete: %.6f seconds\n\n", serial_time);

    printf("[STATUS] Running OpenMP convolution...\n");
    double openmp_start = omp_get_wtime();
    convolve_openmp(input_image, output_openmp, width, height, kernel, num_threads);
    double openmp_end = omp_get_wtime();
    double openmp_time = openmp_end - openmp_start;
    printf("[STATUS] OpenMP complete: %.6f seconds\n\n", openmp_time);

    double speedup = serial_time / openmp_time;
    double throughput = img_size / (openmp_time * 1e6);
    double rmse = calculate_rmse(output_serial, output_openmp, width, height);

    printf("==================== RESULTS ====================\n");
    printf("  Image size       : %d x %d\n", width, height);
    printf("  Total pixels     : %d\n", img_size);
    printf("  Kernel size      : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("  -----------------------------------------\n");
    printf("  Serial CPU time  : %.6f seconds\n", serial_time);
    printf("  OpenMP time      : %.6f seconds\n", openmp_time);
    printf("  Speedup          : %.4f x\n", speedup);
    printf("  Throughput       : %.2f Mpixels/sec\n", throughput);
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

    if (write_pgm(output_filename, output_openmp, width, height, maxval) != 0)
    {
        free(input_image);
        free(output_openmp);
        free(output_serial);
        return EXIT_FAILURE;
    }

    free(input_image);
    free(output_openmp);
    free(output_serial);

    printf("\n[DONE] OpenMP convolution completed successfully.\n");
    return EXIT_SUCCESS;
}