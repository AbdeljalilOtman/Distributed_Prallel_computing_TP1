/*
 * =============================================================================
 * Exercise 3: Block Matrix Multiplication (Cache-Optimized)
 * =============================================================================
 * 
 * WHAT IS BLOCK MATRIX MULTIPLICATION?
 * ------------------------------------
 * Instead of processing individual elements, we divide matrices into smaller
 * "blocks" (or "tiles") of size B×B. We then multiply these blocks together.
 * 
 * WHY DOES BLOCKING HELP?
 * -----------------------
 * The key insight is about CPU cache:
 * 
 * 1. CPU caches are MUCH faster than main memory (100x faster!)
 * 2. Cache is small (L1: ~32KB, L2: ~256KB, L3: ~8MB per core)
 * 3. When we access data, the CPU loads it into cache
 * 4. If we can reuse data while it's still in cache, we win!
 * 
 * With blocking:
 * - We load a B×B block of A into cache
 * - We load a B×B block of B into cache  
 * - We compute ALL products between these blocks
 * - We reuse each element many times before it's evicted!
 * 
 * OPTIMAL BLOCK SIZE:
 * -------------------
 * - Too small: Not enough computation per cache load (overhead)
 * - Too large: Blocks don't fit in cache (cache misses)
 * - Sweet spot: Blocks fit in L1/L2 cache
 * 
 * For doubles (8 bytes), with 32KB L1 cache:
 *   3 blocks × B² × 8 bytes ≤ 32KB
 *   B² ≤ 32KB / 24 ≈ 1365
 *   B ≤ 36 (so B = 32 is a good choice)
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
void multiply_standard(int n, double *A, double *B, double *C);
void multiply_blocked(int n, int block_size, double *A, double *B, double *C);
void initialize_matrices(int n, double *A, double *B, double *C);
double compute_bandwidth(int n, double time_sec);
double compute_gflops(int n, double time_sec);
int verify_results(int n, double *C1, double *C2, double tolerance);

/* -----------------------------------------------------------------------------
 * Main Program
 * -------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    /* Matrix size - use a large matrix to see cache effects clearly */
    int n = 1024;
    
    /* Block sizes to test - finding the optimal one is key! */
    int block_sizes[] = {8, 16, 32, 64, 128, 256};
    int num_blocks = sizeof(block_sizes) / sizeof(block_sizes[0]);
    
    printf("====================================================================\n");
    printf("      Exercise 3: Block Matrix Multiplication Optimization         \n");
    printf("====================================================================\n\n");
    
    printf("Matrix size: %d × %d\n", n, n);
    printf("Memory per matrix: %.2f MB\n", (n * n * sizeof(double)) / (1024.0 * 1024.0));
    printf("\n");

    /* Allocate matrices */
    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));
    double *C_ref = (double *)malloc(n * n * sizeof(double));  /* Reference for verification */
    
    if (!A || !B || !C || !C_ref)
    {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        return 1;
    }
    
    /* Initialize matrices with test data */
    initialize_matrices(n, A, B, C);
    
    /* ----- Baseline: Standard multiplication ----- */
    printf("Running standard (non-blocked) multiplication...\n");
    memset(C_ref, 0, n * n * sizeof(double));
    
    clock_t start = clock();
    multiply_standard(n, A, B, C_ref);
    clock_t end = clock();
    
    double time_standard = (double)(end - start) / CLOCKS_PER_SEC;
    double bw_standard = compute_bandwidth(n, time_standard);
    double gflops_standard = compute_gflops(n, time_standard);
    
    printf("  Time: %.4f seconds\n", time_standard);
    printf("  Bandwidth: %.2f MB/s\n", bw_standard);
    printf("  Performance: %.2f GFLOPS\n\n", gflops_standard);
    
    /* ----- Test different block sizes ----- */
    printf("+------------+-----------+-------------+-----------+----------+----------+\n");
    printf("| Block Size | Time (s)  | BW (MB/s)   | GFLOPS    | Speedup  | Correct? |\n");
    printf("+------------+-----------+-------------+-----------+----------+----------+\n");
    
    double best_time = time_standard;
    int best_block = 0;
    
    for (int b = 0; b < num_blocks; b++)
    {
        int block_size = block_sizes[b];
        
        /* Skip if block size doesn't divide n evenly (for simplicity) */
        if (n % block_size != 0)
        {
            printf("|    %3d     |  skipped (doesn't divide %d evenly)              |\n", 
                   block_size, n);
            continue;
        }
        
        /* Reset result matrix */
        memset(C, 0, n * n * sizeof(double));
        
        /* Time the blocked multiplication */
        start = clock();
        multiply_blocked(n, block_size, A, B, C);
        end = clock();
        
        double time_blocked = (double)(end - start) / CLOCKS_PER_SEC;
        double bw_blocked = compute_bandwidth(n, time_blocked);
        double gflops_blocked = compute_gflops(n, time_blocked);
        double speedup = time_standard / time_blocked;
        
        /* Verify correctness */
        int correct = verify_results(n, C_ref, C, 1e-10);
        
        /* Track best block size */
        if (time_blocked < best_time)
        {
            best_time = time_blocked;
            best_block = block_size;
        }
        
        printf("|    %3d     |  %7.4f  |  %9.2f  |  %7.2f  |  %6.2fx |    %s   |\n",
               block_size, time_blocked, bw_blocked, gflops_blocked, speedup,
               correct ? "YES" : "NO");
    }
    
    printf("+------------+-----------+-------------+-----------+----------+----------+\n");
    
    /* Print analysis */
    printf("\n");
    printf("====================================================================\n");
    printf("                       ANALYSIS & EXPLANATION                       \n");
    printf("====================================================================\n");
    printf("\n");
    
    printf("OPTIMAL BLOCK SIZE: %d\n", best_block);
    printf("---------------------\n");
    printf("The optimal block size of %d gives the best performance because:\n\n", best_block);
    
    printf("1. CACHE FIT:\n");
    printf("   * Three %dx%d blocks = %d doubles = %.1f KB\n", 
           best_block, best_block, 3 * best_block * best_block,
           (3.0 * best_block * best_block * sizeof(double)) / 1024.0);
    printf("   * This fits nicely in L1 cache (~32KB) or L2 cache (~256KB)\n\n");
    
    printf("2. DATA REUSE:\n");
    printf("   * Each element in a block is used %d times before eviction\n", best_block);
    printf("   * This maximizes the computation done per memory access\n\n");
    
    printf("3. MEMORY HIERARCHY:\n");
    printf("   * Small blocks: Too much overhead, not enough work per block\n");
    printf("   * Large blocks: Don't fit in cache, cause cache thrashing\n");
    printf("   * Optimal: Maximum work while data stays in fast cache\n\n");
    
    printf("WHY BLOCKING WORKS (Visual Example):\n");
    printf("------------------------------------\n");
    printf("Standard: C[i][j] = sum(A[i][k] * B[k][j]) for all k\n");
    printf("  -> We touch ALL of A and B for each element of C\n");
    printf("  -> Data gets evicted from cache before we can reuse it\n\n");
    
    printf("Blocked: Process BxB submatrices at a time\n");
    printf("  -> Load small blocks of A, B, C into cache\n");
    printf("  -> Do ALL multiplications between these blocks\n");
    printf("  -> Reuse each loaded value B times!\n\n");
    
    printf("MEMORY BANDWIDTH IMPROVEMENT:\n");
    printf("-----------------------------\n");
    printf("  Standard:  Each element loaded ~n times from memory\n");
    printf("  Blocked:   Each element loaded ~n/B times from memory\n");
    printf("  Speedup ≈  B (the block size) for memory-bound operations\n\n");
    
    /* Clean up */
    free(A);
    free(B);
    free(C);
    free(C_ref);
    
    return 0;
}

/* -----------------------------------------------------------------------------
 * STANDARD MATRIX MULTIPLICATION (for comparison)
 * 
 * Classic i-j-k order with poor cache behavior.
 * -------------------------------------------------------------------------- */
void multiply_standard(int n, double *A, double *B, double *C)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* -----------------------------------------------------------------------------
 * BLOCKED MATRIX MULTIPLICATION
 * 
 * The key optimization! We process the matrices in B×B blocks.
 * 
 * The loop structure is:
 * - ii, jj, kk: Iterate over blocks (stride = block_size)
 * - i, j, k: Iterate within each block
 * 
 * For each block C[ii:ii+B][jj:jj+B]:
 *   We accumulate: A[ii:ii+B][kk:kk+B] × B[kk:kk+B][jj:jj+B]
 * -------------------------------------------------------------------------- */
void multiply_blocked(int n, int block_size, double *A, double *B, double *C)
{
    int BS = block_size;  /* Shorthand for block size (can't use B - it's the matrix!) */
    
    /* Loop over blocks of C */
    for (int ii = 0; ii < n; ii += BS)
    {
        for (int jj = 0; jj < n; jj += BS)
        {
            /* Loop over blocks that contribute to this C block */
            for (int kk = 0; kk < n; kk += BS)
            {
                /* 
                 * Multiply block A[ii:ii+BS, kk:kk+BS] with 
                 * block B[kk:kk+BS, jj:jj+BS] and accumulate into
                 * block C[ii:ii+BS, jj:jj+BS]
                 * 
                 * These three blocks should fit in cache!
                 */
                for (int i = ii; i < ii + BS && i < n; i++)
                {
                    for (int k = kk; k < kk + BS && k < n; k++)
                    {
                        /* Cache A[i][k] - it's used BS times in inner loop */
                        double a_ik = A[i * n + k];
                        
                        for (int j = jj; j < jj + BS && j < n; j++)
                        {
                            /* 
                             * Both B and C are accessed sequentially (row-wise)
                             * within this block - excellent cache behavior!
                             */
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------------
 * MATRIX INITIALIZATION
 * -------------------------------------------------------------------------- */
void initialize_matrices(int n, double *A, double *B, double *C)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            /* Use patterns that won't overflow but are non-trivial */
            A[i * n + j] = (double)((i + j) % 100) / 100.0;
            B[i * n + j] = (double)((i * j + 1) % 100) / 100.0;
            C[i * n + j] = 0.0;
        }
    }
}

/* -----------------------------------------------------------------------------
 * VERIFY RESULTS
 * Compare two result matrices to ensure blocked version is correct.
 * -------------------------------------------------------------------------- */
int verify_results(int n, double *C1, double *C2, double tolerance)
{
    for (int i = 0; i < n * n; i++)
    {
        double diff = C1[i] - C2[i];
        if (diff < 0) diff = -diff;  /* Absolute value */
        if (diff > tolerance)
        {
            return 0;  /* Mismatch found */
        }
    }
    return 1;  /* All elements match */
}

/* -----------------------------------------------------------------------------
 * COMPUTE MEMORY BANDWIDTH
 * -------------------------------------------------------------------------- */
double compute_bandwidth(int n, double time_sec)
{
    if (time_sec <= 0) return 0.0;
    
    double bytes = 3.0 * n * n * sizeof(double);
    double megabytes = bytes / (1024.0 * 1024.0);
    return megabytes / time_sec;
}

/* -----------------------------------------------------------------------------
 * COMPUTE GFLOPS
 * -------------------------------------------------------------------------- */
double compute_gflops(int n, double time_sec)
{
    if (time_sec <= 0) return 0.0;
    
    double flops = 2.0 * (double)n * n * n;
    return flops / (time_sec * 1e9);
}
