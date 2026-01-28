/*
 * =============================================================================
 * Exercise 2: Optimizing Matrix Multiplication
 * =============================================================================
 * 
 * This program demonstrates two approaches to matrix multiplication:
 * 
 * 1. STANDARD VERSION (i-j-k loop order):
 *    - The classic textbook approach
 *    - For each element C[i][j], we compute the dot product of row i of A 
 *      with column j of B
 *    - Problem: Accessing B column-wise causes poor cache performance because
 *      matrices are stored in row-major order in C
 * 
 * 2. OPTIMIZED VERSION (i-k-j loop order):
 *    - Reorders the loops to access both A and B in a cache-friendly manner
 *    - Instead of computing one C[i][j] at a time, we update an entire row of C
 *      using one element from A and one row from B
 *    - This keeps memory access sequential (row-wise), which is how data is
 *      stored in memory, resulting in better cache utilization
 * 
 * WHY DOES LOOP ORDER MATTER?
 * ---------------------------
 * In C, 2D arrays are stored in row-major order:
 *   a[0][0], a[0][1], a[0][2], ..., a[1][0], a[1][1], ...
 * 
 * When we access elements sequentially in memory (stride-1 access), the CPU
 * can prefetch data efficiently. When we jump around (like accessing columns),
 * we get cache misses, which are expensive (100x slower than cache hits).
 * 
 * MEMORY BANDWIDTH CALCULATION:
 * ----------------------------
 * For an n×n matrix multiplication:
 * - Total operations: 2 * n^3 (n^3 multiplications + n^3 additions)
 * - Data accessed: 3 * n^2 * sizeof(double) bytes (matrices A, B, C)
 * - Bandwidth = Data / Time
 * 
 * Author: Student - Parallel and Distributed Computing Course
 * Date: January 2026
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

/* -----------------------------------------------------------------------------
 * Function Prototypes
 * -------------------------------------------------------------------------- */
void multiply_ijk(int n, double *A, double *B, double *C);
void multiply_ikj(int n, double *A, double *B, double *C);
void initialize_matrices(int n, double *A, double *B, double *C);
void print_results(const char *version, int n, double time_sec);
double compute_bandwidth(int n, double time_sec);
double compute_gflops(int n, double time_sec);

/* -----------------------------------------------------------------------------
 * Main Program
 * -------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    /*
     * We use different matrix sizes to see how performance scales.
     * Larger matrices show the cache effects more clearly.
     */
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║     Exercise 2: Matrix Multiplication - Cache Optimization       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("This program compares two loop orderings for matrix multiplication:\n");
    printf("  • Standard (i-j-k): Classic textbook approach\n");
    printf("  • Optimized (i-k-j): Cache-friendly memory access pattern\n\n");
    
    printf("┌────────┬─────────────────────────┬─────────────────────────┬───────────┐\n");
    printf("│  Size  │   Standard (i-j-k)      │   Optimized (i-k-j)     │  Speedup  │\n");
    printf("│  n×n   │  Time(s)   BW(MB/s)     │  Time(s)   BW(MB/s)     │           │\n");
    printf("├────────┼─────────────────────────┼─────────────────────────┼───────────┤\n");
    
    for (int s = 0; s < num_sizes; s++)
    {
        int n = sizes[s];
        
        /* 
         * Allocate matrices as 1D arrays for simplicity and to ensure
         * contiguous memory allocation. We access element (i,j) as A[i*n + j].
         */
        double *A = (double *)malloc(n * n * sizeof(double));
        double *B = (double *)malloc(n * n * sizeof(double));
        double *C = (double *)malloc(n * n * sizeof(double));
        
        if (!A || !B || !C)
        {
            fprintf(stderr, "Error: Memory allocation failed for n=%d\n", n);
            free(A); free(B); free(C);
            continue;
        }
        
        /* Initialize matrices with some values */
        initialize_matrices(n, A, B, C);
        
        /* ----- Test 1: Standard i-j-k multiplication ----- */
        clock_t start_ijk = clock();
        multiply_ijk(n, A, B, C);
        clock_t end_ijk = clock();
        double time_ijk = (double)(end_ijk - start_ijk) / CLOCKS_PER_SEC;
        double bw_ijk = compute_bandwidth(n, time_ijk);
        
        /* Reset C matrix for fair comparison */
        memset(C, 0, n * n * sizeof(double));
        
        /* ----- Test 2: Optimized i-k-j multiplication ----- */
        clock_t start_ikj = clock();
        multiply_ikj(n, A, B, C);
        clock_t end_ikj = clock();
        double time_ikj = (double)(end_ikj - start_ikj) / CLOCKS_PER_SEC;
        double bw_ikj = compute_bandwidth(n, time_ikj);
        
        /* Calculate speedup */
        double speedup = time_ijk / time_ikj;
        
        /* Print results in a nice table format */
        printf("│ %4d   │  %7.4f   %8.2f    │  %7.4f   %8.2f    │   %5.2fx   │\n",
               n, time_ijk, bw_ijk, time_ikj, bw_ikj, speedup);
        
        /* Clean up */
        free(A);
        free(B);
        free(C);
    }
    
    printf("└────────┴─────────────────────────┴─────────────────────────┴───────────┘\n");
    
    /* Print explanation */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                        ANALYSIS & EXPLANATION                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("WHY IS THE i-k-j VERSION FASTER?\n");
    printf("─────────────────────────────────\n");
    printf("1. MEMORY ACCESS PATTERN:\n");
    printf("   • In i-j-k: B is accessed column-wise (B[k][j] with k varying)\n");
    printf("     This jumps n elements in memory between accesses = BAD for cache\n");
    printf("\n");
    printf("   • In i-k-j: Both A and B are accessed row-wise (stride-1)\n");
    printf("     Elements are adjacent in memory = GOOD for cache\n");
    printf("\n");
    printf("2. CACHE BEHAVIOR:\n");
    printf("   • CPU cache loads data in 'cache lines' (typically 64 bytes)\n");
    printf("   • Row-wise access uses all data in a cache line before eviction\n");
    printf("   • Column-wise access wastes most of each cache line\n");
    printf("\n");
    printf("3. PREFETCHING:\n");
    printf("   • CPU can predict sequential memory access and prefetch data\n");
    printf("   • Random/strided access defeats prefetching\n");
    printf("\n");
    printf("BANDWIDTH FORMULA:\n");
    printf("──────────────────\n");
    printf("  Bandwidth (MB/s) = (3 × n² × sizeof(double)) / (time × 1024²)\n");
    printf("  Where 3 matrices (A, B, C) of n×n doubles are accessed.\n");
    printf("\n");
    
    return 0;
}

/* -----------------------------------------------------------------------------
 * STANDARD MATRIX MULTIPLICATION (i-j-k loop order)
 * 
 * This is the classic textbook algorithm:
 *   C[i][j] = sum over k of A[i][k] * B[k][j]
 * 
 * Problem: When we loop over k in the innermost loop, we access B column-wise.
 * In row-major storage, this means we jump n elements between each access,
 * causing frequent cache misses.
 * -------------------------------------------------------------------------- */
void multiply_ijk(int n, double *A, double *B, double *C)
{
    for (int i = 0; i < n; i++)           /* For each row of C */
    {
        for (int j = 0; j < n; j++)       /* For each column of C */
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)   /* Dot product of row i and col j */
            {
                /* 
                 * A[i][k] - accessed row-wise (good!)
                 * B[k][j] - accessed column-wise (bad! jumps n elements each time)
                 */
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* -----------------------------------------------------------------------------
 * OPTIMIZED MATRIX MULTIPLICATION (i-k-j loop order)
 * 
 * Key insight: Reorder loops so both A and B are accessed row-wise.
 * 
 * Instead of computing C[i][j] completely before moving on, we:
 * - Take one element A[i][k]
 * - Multiply it with an entire row of B (B[k][0], B[k][1], ...)
 * - Add the results to the corresponding row of C
 * 
 * This way, we access B row-wise, which matches how it's stored in memory!
 * -------------------------------------------------------------------------- */
void multiply_ikj(int n, double *A, double *B, double *C)
{
    for (int i = 0; i < n; i++)           /* For each row of A/C */
    {
        for (int k = 0; k < n; k++)       /* For each element in row i of A */
        {
            double a_ik = A[i * n + k];   /* Cache this value - used n times */
            
            for (int j = 0; j < n; j++)   /* Update entire row i of C */
            {
                /* 
                 * B[k][j] - accessed row-wise (good! sequential access)
                 * C[i][j] - accessed row-wise (good! sequential access)
                 */
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

/* -----------------------------------------------------------------------------
 * MATRIX INITIALIZATION
 * 
 * We initialize A and B with simple patterns:
 * - A: Hilbert-like matrix (1/(i+j+1) scaled)
 * - B: Simple pattern based on position
 * - C: Zero matrix (will hold the result)
 * -------------------------------------------------------------------------- */
void initialize_matrices(int n, double *A, double *B, double *C)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            A[i * n + j] = 1.0 / (i + j + 1);    /* Hilbert-like */
            B[i * n + j] = (double)(i + j);      /* Simple pattern */
            C[i * n + j] = 0.0;                   /* Result matrix */
        }
    }
}

/* -----------------------------------------------------------------------------
 * COMPUTE MEMORY BANDWIDTH
 * 
 * Bandwidth measures how fast we're moving data between memory and CPU.
 * For matrix multiplication:
 * - We read all of A: n² doubles
 * - We read all of B: n² doubles  
 * - We write all of C: n² doubles
 * - Total: 3 × n² × 8 bytes (double = 8 bytes)
 * -------------------------------------------------------------------------- */
double compute_bandwidth(int n, double time_sec)
{
    if (time_sec <= 0) return 0.0;
    
    double bytes = 3.0 * n * n * sizeof(double);
    double megabytes = bytes / (1024.0 * 1024.0);
    return megabytes / time_sec;
}

/* -----------------------------------------------------------------------------
 * COMPUTE GFLOPS (Giga Floating Point Operations per Second)
 * 
 * Matrix multiplication performs:
 * - n³ multiplications
 * - n³ additions
 * - Total: 2 × n³ floating point operations
 * -------------------------------------------------------------------------- */
double compute_gflops(int n, double time_sec)
{
    if (time_sec <= 0) return 0.0;
    
    double flops = 2.0 * (double)n * n * n;
    return flops / (time_sec * 1e9);
}

