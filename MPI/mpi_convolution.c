/*
 * =============================================================================
 * MPI Image Convolution (Distributed Memory Parallelism)
 * EC7207 - High Performance Computing
 * =============================================================================
 *
 * Description:
 *   Performs 2D convolution on a grayscale PGM image using MPI.
 *   The image is split row-wise among MPI processes. Halo (ghost) rows are
 *   exchanged between neighboring processes to handle boundary pixels.
 *   Results are compared against the serial baseline using RMSE.
 *
 * Compilation:
 *   mpicc -O2 -o mpi_conv mpi_convolution.c -lm
 *
 * Usage:
 *   mpirun -np <num_procs> ./mpi_conv <input.pgm> <output.pgm>
 *   mpirun -np <num_procs> ./mpi_conv --generate <width> <height> <output.pgm>
 *
 * Example:
 *   mpirun -np 4 ./mpi_conv input.pgm output_mpi.pgm
 *
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ========================= Configuration ========================= */

#define KERNEL_SIZE 3
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

/* Gaussian Blur 3x3 Kernel */
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

/**
 * Serial convolution for baseline comparison (run on rank 0 only).
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

/* ========================= Local Convolution ========================= */

/**
 * Perform convolution on a local block (with halo rows included).
 *
 * local_data: local rows + halo rows (total_local_rows = local_rows + 2*KERNEL_RADIUS)
 * output: result for only the local_rows (without halo)
 * width: image width
 * local_rows: number of actual rows this process owns
 * total_rows: total rows in local_data including halos
 */
void convolve_local(const unsigned char *local_data, unsigned char *output,
                    int width, int local_rows, int total_rows,
                    float kernel[KERNEL_SIZE][KERNEL_SIZE])
{
    /*
     * local_data layout:
     *   [halo_top: KERNEL_RADIUS rows]  (may be zeros if first process)
     *   [actual rows: local_rows]
     *   [halo_bottom: KERNEL_RADIUS rows]  (may be zeros if last process)
     *
     * We compute convolution only for the actual rows (offset by KERNEL_RADIUS).
     */
    for (int row = 0; row < local_rows; row++)
    {
        int data_row = row + KERNEL_RADIUS; /* Row in local_data */

        for (int col = 0; col < width; col++)
        {
            float sum = 0.0f;

            for (int ki = -KERNEL_RADIUS; ki <= KERNEL_RADIUS; ki++)
            {
                for (int kj = -KERNEL_RADIUS; kj <= KERNEL_RADIUS; kj++)
                {
                    int ni = data_row + ki;
                    int nj = col + kj;

                    /* Boundary check within local block */
                    if (ni >= 0 && ni < total_rows && nj >= 0 && nj < width)
                    {
                        sum += local_data[ni * width + nj] *
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

/* ========================= RMSE Calculation ========================= */

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

int main(int argc, char *argv[])
{
    int rank, num_procs;
    int width, height, maxval = 255;
    unsigned char *full_image = NULL;
    unsigned char *output_mpi = NULL;
    unsigned char *output_serial = NULL;
    const char *output_filename = NULL;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0)
    {
        printf("============================================\n");
        printf("  MPI Image Convolution (Distributed Memory)\n");
        printf("  EC7207 - High Performance Computing\n");
        printf("============================================\n\n");
    }

    /* Parse command-line arguments on rank 0 */
    if (rank == 0)
    {
        if (argc >= 3 && strcmp(argv[1], "--generate") != 0)
        {
            full_image = read_pgm(argv[1], &width, &height, &maxval);
            if (!full_image)
            {
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            output_filename = argv[2];
            printf("[INFO] Read PGM image: %d x %d\n", width, height);
        }
        else if (argc >= 5 && strcmp(argv[1], "--generate") == 0)
        {
            width = atoi(argv[2]);
            height = atoi(argv[3]);
            if (width <= 0 || height <= 0)
            {
                fprintf(stderr, "Error: Invalid dimensions\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            full_image = generate_test_image(width, height);
            if (!full_image)
            {
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            output_filename = argv[4];
            printf("[INFO] Generated test image: %d x %d\n", width, height);
        }
        else
        {
            if (rank == 0)
            {
                printf("Usage:\n");
                printf("  mpirun -np <N> %s <input.pgm> <output.pgm>\n", argv[0]);
                printf("  mpirun -np <N> %s --generate <W> <H> <output.pgm>\n", argv[0]);
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        printf("[CONFIG] Image size    : %d x %d (%d pixels)\n", width, height, width * height);
        printf("[CONFIG] MPI processes : %d\n", num_procs);
        printf("[CONFIG] Kernel size   : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
        printf("[CONFIG] Kernel type   : Gaussian Blur\n\n");
    }

    /* Broadcast image dimensions to all processes */
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxval, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Normalize kernel on all processes */
    normalize_kernel(gaussian_kernel);

    /* =========== Calculate Row Distribution =========== */
    /*
     * Distribute rows as evenly as possible.
     * If height is not perfectly divisible, first (height % num_procs) processes
     * get one extra row.
     */
    int *rows_per_proc = (int *)malloc(num_procs * sizeof(int));
    int *row_offsets = (int *)malloc(num_procs * sizeof(int));

    int base_rows = height / num_procs;
    int remainder = height % num_procs;

    int offset = 0;
    for (int p = 0; p < num_procs; p++)
    {
        rows_per_proc[p] = base_rows + (p < remainder ? 1 : 0);
        row_offsets[p] = offset;
        offset += rows_per_proc[p];
    }

    int my_rows = rows_per_proc[rank];
    int my_offset = row_offsets[rank];

    if (rank == 0)
    {
        printf("[DIST] Row distribution:\n");
        for (int p = 0; p < num_procs; p++)
        {
            printf("  Rank %d: rows %d-%d (%d rows)\n",
                   p, row_offsets[p], row_offsets[p] + rows_per_proc[p] - 1,
                   rows_per_proc[p]);
        }
        printf("\n");
    }

    /* =========== Scatter Image Data with Halo Rows =========== */
    /*
     * Each process needs:
     *   - Its own rows (my_rows)
     *   - KERNEL_RADIUS halo rows from the top neighbor
     *   - KERNEL_RADIUS halo rows from the bottom neighbor
     *
     * Total local buffer size = (my_rows + 2 * KERNEL_RADIUS) * width
     */
    int total_local_rows = my_rows + 2 * KERNEL_RADIUS;
    unsigned char *local_data = (unsigned char *)calloc(total_local_rows * width,
                                                        sizeof(unsigned char));
    unsigned char *local_output = (unsigned char *)malloc(my_rows * width *
                                                          sizeof(unsigned char));

    /* Rank 0 sends data to each process using point-to-point communication */
    if (rank == 0)
    {
        /* Copy own rows + prepare halo */
        for (int r = 0; r < total_local_rows; r++)
        {
            int global_row = my_offset - KERNEL_RADIUS + r;
            if (global_row >= 0 && global_row < height)
            {
                memcpy(&local_data[r * width],
                       &full_image[global_row * width],
                       width * sizeof(unsigned char));
            }
            /* Otherwise, zero-padded (calloc) */
        }

        /* Send to other processes */
        for (int p = 1; p < num_procs; p++)
        {
            int p_rows = rows_per_proc[p];
            int p_offset = row_offsets[p];
            int p_total_rows = p_rows + 2 * KERNEL_RADIUS;

            unsigned char *send_buf = (unsigned char *)calloc(p_total_rows * width,
                                                              sizeof(unsigned char));
            for (int r = 0; r < p_total_rows; r++)
            {
                int global_row = p_offset - KERNEL_RADIUS + r;
                if (global_row >= 0 && global_row < height)
                {
                    memcpy(&send_buf[r * width],
                           &full_image[global_row * width],
                           width * sizeof(unsigned char));
                }
            }

            MPI_Send(send_buf, p_total_rows * width, MPI_UNSIGNED_CHAR,
                     p, 0, MPI_COMM_WORLD);
            free(send_buf);
        }
    }
    else
    {
        /* Receive local data (including halo rows) from rank 0 */
        MPI_Recv(local_data, total_local_rows * width, MPI_UNSIGNED_CHAR,
                 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* =========== Perform Convolution with Timing =========== */
    double mpi_start = MPI_Wtime();

    convolve_local(local_data, local_output, width, my_rows, total_local_rows,
                   gaussian_kernel);

    MPI_Barrier(MPI_COMM_WORLD);
    double mpi_end = MPI_Wtime();
    double mpi_time = mpi_end - mpi_start;

    /* Get max time across all processes */
    double max_mpi_time;
    MPI_Reduce(&mpi_time, &max_mpi_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* =========== Gather Results on Rank 0 =========== */
    if (rank == 0)
    {
        output_mpi = (unsigned char *)malloc(width * height * sizeof(unsigned char));

        /* Copy own results */
        memcpy(&output_mpi[my_offset * width], local_output,
               my_rows * width * sizeof(unsigned char));

        /* Receive from other processes */
        for (int p = 1; p < num_procs; p++)
        {
            MPI_Recv(&output_mpi[row_offsets[p] * width],
                     rows_per_proc[p] * width, MPI_UNSIGNED_CHAR,
                     p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        /* Send results to rank 0 */
        MPI_Send(local_output, my_rows * width, MPI_UNSIGNED_CHAR,
                 0, 1, MPI_COMM_WORLD);
    }

    /* =========== Serial Baseline & Analysis (Rank 0 only) =========== */
    if (rank == 0)
    {
        /* Run serial baseline for comparison */
        output_serial = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        printf("[STATUS] Running serial baseline for comparison...\n");

        double serial_start = MPI_Wtime();
        convolve_serial(full_image, output_serial, width, height, gaussian_kernel);
        double serial_end = MPI_Wtime();
        double serial_time = serial_end - serial_start;

        /* Performance metrics */
        double speedup = serial_time / max_mpi_time;
        double efficiency = speedup / num_procs;
        double rmse = calculate_rmse(output_serial, output_mpi, width, height);

        printf("\n==================== RESULTS ====================\n");
        printf("  Image size       : %d x %d\n", width, height);
        printf("  Total pixels     : %d\n", width * height);
        printf("  Kernel size      : %d x %d\n", KERNEL_SIZE, KERNEL_SIZE);
        printf("  MPI processes    : %d\n", num_procs);
        printf("  -----------------------------------------\n");
        printf("  Serial time      : %.6f seconds\n", serial_time);
        printf("  MPI time         : %.6f seconds\n", max_mpi_time);
        printf("  Speedup          : %.4f x\n", speedup);
        printf("  Efficiency       : %.4f (%.1f%%)\n", efficiency, efficiency * 100.0);
        printf("  Throughput       : %.2f Mpixels/sec\n",
               (width * height) / (max_mpi_time * 1e6));
        printf("  RMSE             : %.6f\n", rmse);
        printf("=================================================\n\n");

        if (rmse < 1e-6)
        {
            printf("[VERIFY] PASSED - Output matches serial baseline exactly.\n");
        }
        else if (rmse < 1.0)
        {
            printf("[VERIFY] PASSED - Output within acceptable tolerance.\n");
        }
        else
        {
            printf("[VERIFY] WARNING - Significant difference from serial baseline!\n");
        }

        /* Write output */
        write_pgm(output_filename, output_mpi, width, height, maxval);
        printf("[INFO] Wrote output: %s\n", output_filename);

        /* Cleanup rank 0 */
        free(full_image);
        free(output_mpi);
        free(output_serial);
    }

    /* Cleanup all ranks */
    free(local_data);
    free(local_output);
    free(rows_per_proc);
    free(row_offsets);

    if (rank == 0)
    {
        printf("\n[DONE] MPI convolution completed successfully.\n");
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
