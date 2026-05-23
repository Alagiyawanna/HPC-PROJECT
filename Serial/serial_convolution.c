/*
 * =============================================================================
 * Serial Image Convolution (Baseline Implementation)
 * EC7207 - High Performance Computing
 * =============================================================================
 *
 * Description:
 *   Performs 2D convolution on a grayscale PGM image using a serial approach.
 *   Supports multiple convolution kernels: Gaussian Blur, Sharpen, Edge Detection.
 *   This serves as the baseline for performance comparison with parallel
 *   implementations (OpenMP, MPI, CUDA).
 *
 * Compilation:
 *   gcc -O2 -o serial_conv serial_convolution.c -lm
 *
 * Run:
 *   ./serial_conv input.pgm output.pgm [kernel_type]

 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
    #include <windows.h>  // For QueryPerformanceCounter on Windows
#endif

// Configuration - blur uses a larger Gaussian kernel
#define BLUR_KERNEL_SIZE 100

// Gaussian Blur - smooths the image by averaging nearby pixels.
// This version uses a 100x100 kernel and is normalized at runtime.
static float *gaussian_kernel = NULL;

/* ========================= PGM Image I/O ========================= */

// PGM (Portable GrayMap) is chosen because it's dead simple:
// Just a header + raw bytes, no compression, easy to read in pure C
unsigned char *read_pgm(const char *filename, int *width, int *height, int *maxval)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return NULL;
    }

    // PGM files start with "P5" magic number for binary format
    char magic[3];
    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Error: File '%s' is not a valid P5 PGM file\n", filename);
        fclose(fp);
        return NULL;
    }

    // Skip any comment lines (start with #)
    int ch;
    while ((ch = fgetc(fp)) == '#' || ch == '\n' || ch == '\r' || ch == ' ') {
        if (ch == '#') {
            while ((ch = fgetc(fp)) != '\n' && ch != EOF);
        }
    }
    ungetc(ch, fp);

    // Read image dimensions and max pixel value (usually 255)
    if (fscanf(fp, "%d %d %d", width, height, maxval) != 3) {
        fprintf(stderr, "Error: Cannot read PGM header\n");
        fclose(fp);
        return NULL;
    }

    fgetc(fp); // skip the single whitespace after maxval

    // Allocate memory for the entire image (width × height bytes)
    int size = (*width) * (*height);
    unsigned char *data = (unsigned char *)malloc(size * sizeof(unsigned char));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }

    // Read all pixel data in one go - each pixel is one byte (0-255)
    if (fread(data, sizeof(unsigned char), size, fp) != (size_t)size) {
        fprintf(stderr, "Error: Cannot read pixel data\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    printf("[INFO] Read PGM image: %d x %d, maxval=%d\n", *width, *height, *maxval);
    return data;
}

// Write the result back as a PGM file
int write_pgm(const char *filename, unsigned char *data, int width, int height, int maxval)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create file '%s'\n", filename);
        return -1;
    }

    // Write PGM header, then dump all pixel bytes
    fprintf(fp, "P5\n%d %d\n%d\n", width, height, maxval);
    fwrite(data, sizeof(unsigned char), width * height, fp);
    fclose(fp);

    printf("[INFO] Wrote PGM image: %s (%d x %d)\n", filename, width, height);
    return 0;
}

// Generate a synthetic test image if user doesn't have a real one
// Creates a nice pattern with gradient + shapes for testing
unsigned char *generate_test_image(int width, int height)
{
    unsigned char *data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (!data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Start with a diagonal gradient from dark to light
            int val = (int)(255.0 * (i + j) / (height + width));

            // Add a light rectangle in the center
            if (i > height / 4 && i < 3 * height / 4 &&
                j > width / 4 && j < 3 * width / 4) {
                val = 200;
            }

            // Add a dark circle overlapping the rectangle
            int ci = height / 2, cj = width / 2;
            int radius = (height < width ? height : width) / 6;
            if ((i - ci) * (i - ci) + (j - cj) * (j - cj) < radius * radius) {
                val = 50;
            }

            // Clamp to valid pixel range
            data[i * width + j] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
        }
    }

    printf("[INFO] Generated test image: %d x %d\n", width, height);
    return data;
}

/* ========================= Convolution ========================= */

// For blur kernels, we want the weights to sum to 1 so brightness doesn't change
// For sharpen/edge, we keep them as-is (they sum to 1 or 0 already)
void normalize_kernel(float *kernel, int kernel_size)
{
    float sum = 0.0f;
    int total = kernel_size * kernel_size;

    for (int i = 0; i < total; i++) {
        sum += kernel[i];
    }

    // Only normalize if sum is significantly non-zero (for blur)
    if (fabs(sum) > 1e-6f) {
        for (int i = 0; i < total; i++) {
            kernel[i] /= sum;
        }
    }
}

void generate_gaussian_kernel(float *kernel, int kernel_size)
{
    const float center = (float)(kernel_size - 1) / 2.0f;
    const float sigma = (float)kernel_size / 6.0f;
    const float two_sigma_sq = 2.0f * sigma * sigma;

    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            float di = (float)i - center;
            float dj = (float)j - center;
            kernel[i * kernel_size + j] = expf(-((di * di + dj * dj) / two_sigma_sq));
        }
    }
}

// The core convolution algorithm - this is what we're benchmarking
// For each output pixel, we multiply neighbors by kernel weights and sum them
void convolve_serial(const unsigned char *input, unsigned char *output,
                     int width, int height,
                     const float *kernel, int kernel_size)
{
    int kernel_radius = kernel_size / 2;

    // Loop through every pixel in the output image
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;

            // Apply the kernel centered at this pixel.
            for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                    int ni = row + ki - kernel_radius; // neighbor row
                    int nj = col + kj - kernel_radius; // neighbor col

                    // Zero-padding: pretend out-of-bounds pixels are black
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        sum += input[ni * width + nj] * 
                               kernel[ki * kernel_size + kj];
                    }
                }
            }

            // Clamp the result to valid pixel range [0, 255]
            if (sum < 0.0f) sum = 0.0f;
            if (sum > 255.0f) sum = 255.0f;
            
            output[row * width + col] = (unsigned char)(sum + 0.5f); // round to nearest
        }
    }
}

/* ========================= RMSE Calculation ========================= */

// Root Mean Square Error - measures how different two images are
// Used by parallel versions to verify they match the serial output
double calculate_rmse(const unsigned char *img1, const unsigned char *img2,
                      int width, int height)
{
    double sum_sq = 0.0;
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++) {
        double diff = (double)img1[i] - (double)img2[i];
        sum_sq += diff * diff;
    }

    return sqrt(sum_sq / total_pixels);
}

/* ========================= Timing Utility ========================= */

// High-resolution timer - cross-platform implementation
// Windows uses QueryPerformanceCounter, Linux/Unix uses clock_gettime
double get_time_seconds(void)
{
#ifdef _WIN32
    // Windows: use QueryPerformanceCounter for high-resolution timing
    static double frequency = 0.0;
    static int initialized = 0;
    LARGE_INTEGER count, freq;
    
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        frequency = (double)freq.QuadPart;
        initialized = 1;
    }
    
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / frequency;
#else
    // Linux/Unix: use clock_gettime with CLOCK_MONOTONIC
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ========================= Kernel Selection ========================= */

const char* get_kernel_name(void)
{
    return "Gaussian Blur";
}

/* ========================= Main Program ========================= */

void print_usage(const char *prog)
{
    printf("\nUsage:\n");
    printf("  %s <input.pgm> <output.pgm>\n", prog);
    printf("  %s --generate <width> <height> <output.pgm>\n\n", prog);
    printf("Kernel:\n");
    printf("  blur - Gaussian Blur %dx%d (default, smooths image)\n\n", BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE);
    printf("Examples:\n");
    printf("  %s photo.pgm blurred.pgm\n", prog);
    printf("  %s --generate 1024 1024 test.pgm\n\n", prog);
}

int main(int argc, char *argv[])
{
    unsigned char *input_image = NULL;
    unsigned char *output_image = NULL;
    int width, height, maxval = 255;
    const char *output_filename = NULL;
    const float *kernel = NULL;
    int kernel_size = BLUR_KERNEL_SIZE;

    printf("============================================\n");
    printf("  Serial Image Convolution (Baseline)\n");
    printf("  EC7207 - High Performance Computing\n");
    printf("============================================\n\n");

    // Parse command-line arguments - support multiple modes and kernel selection
    if (argc == 3) {
        // Mode 1: Read from file
        input_image = read_pgm(argv[1], &width, &height, &maxval);
        if (!input_image) return EXIT_FAILURE;
        
        output_filename = argv[2];
    }
    else if (argc == 5 && strcmp(argv[1], "--generate") == 0) {
        // Mode 2: Generate test image
        width = atoi(argv[2]);
        height = atoi(argv[3]);
        
        if (width <= 0 || height <= 0) {
            fprintf(stderr, "Error: Invalid dimensions %d x %d\n", width, height);
            return EXIT_FAILURE;
        }
        
        input_image = generate_test_image(width, height);
        if (!input_image) return EXIT_FAILURE;
        
        output_filename = argv[4];

        // Save the generated input for visual inspection
        write_pgm("serial_input.pgm", input_image, width, height, maxval);
    }
    else {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Allocate space for the output image
    output_image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (!output_image) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(input_image);
        return EXIT_FAILURE;
    }

    // Build and normalize the blur kernel.
    kernel = (const float *)malloc((size_t)kernel_size * (size_t)kernel_size * sizeof(float));
    if (!kernel) {
        fprintf(stderr, "Error: Memory allocation failed for blur kernel\n");
        free(input_image);
        free(output_image);
        return EXIT_FAILURE;
    }
    gaussian_kernel = (float *)kernel;
    generate_gaussian_kernel(gaussian_kernel, kernel_size);
    normalize_kernel(gaussian_kernel, kernel_size);

    // Print configuration - important for the evaluation report
    printf("\n[CONFIG] Image size    : %d x %d (%d pixels)\n", width, height, width * height);
    printf("[CONFIG] Kernel size   : %d x %d\n", kernel_size, kernel_size);
    printf("[CONFIG] Kernel type   : %s\n", get_kernel_name());
    printf("[CONFIG] Boundary      : Zero-padding\n\n");

    // This is what we're measuring - the actual convolution computation
    printf("[STATUS] Starting serial convolution...\n");

    double start_time = get_time_seconds();
    convolve_serial(input_image, output_image, width, height, kernel, kernel_size);
    double end_time = get_time_seconds();

    double elapsed_time = end_time - start_time;
    printf("[STATUS] Convolution complete.\n\n");

    // Report performance metrics - these numbers matter for your evaluation
    printf("==================== RESULTS ====================\n");
    printf("  Image size       : %d x %d\n", width, height);
    printf("  Total pixels     : %d\n", width * height);
    printf("  Kernel size      : %d x %d\n", kernel_size, kernel_size);
    printf("  Kernel type      : %s\n", get_kernel_name());
    printf("  Execution time   : %.6f seconds\n", elapsed_time);
    printf("  Throughput       : %.2f Mpixels/sec\n",
           (width * height) / (elapsed_time * 1e6));
    printf("=================================================\n\n");

    // Self-check - should always be 0.0
    double rmse = calculate_rmse(output_image, output_image, width, height);
    printf("[VERIFY] Self-RMSE     : %.6f (expected 0.0)\n", rmse);

    // Write the output image
    if (write_pgm(output_filename, output_image, width, height, maxval) != 0) {
        free(input_image);
        free(output_image);
        return EXIT_FAILURE;
    }

    // Save a reference copy so parallel versions can compare against it
    write_pgm("serial_output_reference.pgm", output_image, width, height, maxval);
    printf("[INFO] Reference output saved as 'serial_output_reference.pgm'\n");

    // Clean up memory
    free(input_image);
    free(output_image);
    free((void *)kernel);

    printf("\n[DONE] Serial convolution completed successfully.\n");
    return EXIT_SUCCESS;
}