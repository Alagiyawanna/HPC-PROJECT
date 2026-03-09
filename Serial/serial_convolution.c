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
 * Usage:
 *   ./serial_conv <input.pgm> <output.pgm> [kernel_type]
 *   ./serial_conv --generate <width> <height> <output.pgm> [kernel_type]
 *
 *   kernel_type options:
 *     blur     - Gaussian Blur 3x3 (default, smooths the image)
 *     sharpen  - Sharpen 3x3 (enhances edges and details)
 *     edge     - Edge Detection 3x3 (Laplacian, highlights boundaries)
 *
 * Examples:
 *   ./serial_conv input.pgm output.pgm blur
 *   ./serial_conv --generate 1024 1024 test.pgm edge
 *
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

// Configuration - we define three different convolution filters that produce different effects
#define KERNEL_SIZE 3 
#define KERNEL_RADIUS (KERNEL_SIZE / 2) 

// Kernel type enumeration - makes code more readable than using magic numbers
typedef enum {
    KERNEL_BLUR,
    KERNEL_SHARPEN,
    KERNEL_EDGE
} KernelType;

// Each kernel does something different to the image:

// Gaussian Blur - smooths the image by averaging nearby pixels
// Higher weight in center, lower on edges creates natural blur
// We'll normalize this at runtime so all weights sum to 1
static float gaussian_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1.0f, 2.0f, 1.0f},
    {2.0f, 4.0f, 2.0f},
    {1.0f, 2.0f, 1.0f}
};

// Sharpen - makes edges crisper by emphasizing differences
// Center is positive, neighbors are negative - enhances contrast
static float sharpen_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    { 0.0f, -1.0f,  0.0f},
    {-1.0f,  5.0f, -1.0f},
    { 0.0f, -1.0f,  0.0f}
};

// Edge Detection (Laplacian) - finds boundaries in the image
// Highlights areas where pixel values change rapidly (edges)
static float edge_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {-1.0f, -1.0f, -1.0f},
    {-1.0f,  8.0f, -1.0f},
    {-1.0f, -1.0f, -1.0f}
};

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
void normalize_kernel(float kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    float sum = 0.0f;
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            sum += kernel[i][j];
        }
    }

    // Only normalize if sum is significantly non-zero (for blur)
    if (fabs(sum) > 1e-6) {
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                kernel[i][j] /= sum;
            }
        }
    }
}

// The core convolution algorithm - this is what we're benchmarking
// For each output pixel, we multiply neighbors by kernel weights and sum them
void convolve_serial(const unsigned char *input, unsigned char *output,
                     int width, int height,
                     float kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    // Loop through every pixel in the output image
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;

            // Apply the 3x3 kernel centered at this pixel
            for (int ki = -KERNEL_RADIUS; ki <= KERNEL_RADIUS; ki++) {
                for (int kj = -KERNEL_RADIUS; kj <= KERNEL_RADIUS; kj++) {
                    int ni = row + ki; // neighbor row
                    int nj = col + kj; // neighbor col

                    // Zero-padding: pretend out-of-bounds pixels are black
                    if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                        sum += input[ni * width + nj] * 
                               kernel[ki + KERNEL_RADIUS][kj + KERNEL_RADIUS];
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

// Parse the kernel name from command line and return the right kernel
const char* get_kernel_name(KernelType type)
{
    switch(type) {
        case KERNEL_BLUR:    return "Gaussian Blur";
        case KERNEL_SHARPEN: return "Sharpen";
        case KERNEL_EDGE:    return "Edge Detection";
        default:             return "Unknown";
    }
}

KernelType parse_kernel_type(const char *str)
{
    if (!str || strcmp(str, "blur") == 0) {
        return KERNEL_BLUR;  // default
    } else if (strcmp(str, "sharpen") == 0) {
        return KERNEL_SHARPEN;
    } else if (strcmp(str, "edge") == 0) {
        return KERNEL_EDGE;
    } else {
        fprintf(stderr, "Warning: Unknown kernel '%s', using blur\n", str);
        return KERNEL_BLUR;
    }
}

float (*get_kernel(KernelType type))[KERNEL_SIZE]
{
    switch(type) {
        case KERNEL_BLUR:    return gaussian_kernel;
        case KERNEL_SHARPEN: return sharpen_kernel;
        case KERNEL_EDGE:    return edge_kernel;
        default:             return gaussian_kernel;
    }
}

/* ========================= Main Program ========================= */

void print_usage(const char *prog)
{
    printf("\nUsage:\n");
    printf("  %s <input.pgm> <output.pgm> [kernel_type]\n", prog);
    printf("  %s --generate <width> <height> <output.pgm> [kernel_type]\n\n", prog);
    printf("Kernel types:\n");
    printf("  blur     - Gaussian Blur (default, smooths image)\n");
    printf("  sharpen  - Sharpen (enhances edges)\n");
    printf("  edge     - Edge Detection (Laplacian)\n\n");
    printf("Examples:\n");
    printf("  %s photo.pgm blurred.pgm blur\n", prog);
    printf("  %s --generate 1024 1024 test.pgm edge\n\n", prog);
}

int main(int argc, char *argv[])
{
    unsigned char *input_image = NULL;
    unsigned char *output_image = NULL;
    int width, height, maxval = 255;
    const char *output_filename = NULL;
    KernelType kernel_type = KERNEL_BLUR; // default

    printf("============================================\n");
    printf("  Serial Image Convolution (Baseline)\n");
    printf("  EC7207 - High Performance Computing\n");
    printf("============================================\n\n");

    // Parse command-line arguments - support multiple modes and kernel selection
    if (argc >= 3 && argc <= 4) {
        // Mode 1: Read from file
        input_image = read_pgm(argv[1], &width, &height, &maxval);
        if (!input_image) return EXIT_FAILURE;
        
        output_filename = argv[2];
        
        // Optional kernel type
        if (argc == 4) {
            kernel_type = parse_kernel_type(argv[3]);
        }
    }
    else if (argc >= 5 && argc <= 6 && strcmp(argv[1], "--generate") == 0) {
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
        
        // Optional kernel type
        if (argc == 6) {
            kernel_type = parse_kernel_type(argv[5]);
        }

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

    // Get the selected kernel and normalize it if needed (blur only)
    float (*kernel)[KERNEL_SIZE] = get_kernel(kernel_type);
    if (kernel_type == KERNEL_BLUR) {
        normalize_kernel(kernel);
    }

    // Print configuration - important for the evaluation report
    printf("\n[CONFIG] Image size    : %d x %d (%d pixels)\n", width, height, width * height);
    printf("[CONFIG] Kernel size   : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("[CONFIG] Kernel type   : %s\n", get_kernel_name(kernel_type));
    printf("[CONFIG] Boundary      : Zero-padding\n\n");

    // This is what we're measuring - the actual convolution computation
    printf("[STATUS] Starting serial convolution...\n");

    double start_time = get_time_seconds();
    convolve_serial(input_image, output_image, width, height, kernel);
    double end_time = get_time_seconds();

    double elapsed_time = end_time - start_time;
    printf("[STATUS] Convolution complete.\n\n");

    // Report performance metrics - these numbers matter for your evaluation
    printf("==================== RESULTS ====================\n");
    printf("  Image size       : %d x %d\n", width, height);
    printf("  Total pixels     : %d\n", width * height);
    printf("  Kernel size      : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
    printf("  Kernel type      : %s\n", get_kernel_name(kernel_type));
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

    printf("\n[DONE] Serial convolution completed successfully.\n");
    return EXIT_SUCCESS;
}