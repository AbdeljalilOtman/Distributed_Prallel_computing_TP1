# TP1 - Optimizing Memory Access
## Parallel and Distributed Computing Course

This folder contains all exercises for TP1, focusing on understanding how memory access patterns affect performance.

---

## 📁 Files Overview

| File | Description |
|------|-------------|
| `exercise1.c` | Impact of memory access stride |
| `exercise2.c` | Matrix multiplication with loop order optimization |
| `exercise3_mxm_bloc.c` | Block matrix multiplication |
| `exercise4_memory_debug.c` | Memory management and Valgrind debugging |
| `exercise5_hpl_benchmark.c` | HPL-style benchmark analysis |
| `TP1_Analysis.ipynb` | Jupyter notebook for plotting and analysis |
| `resultsO0.txt` | Results from Exercise 1 without optimization |
| `resultsO2.txt` | Results from Exercise 1 with -O2 optimization |

---

## 🔧 Compilation & Execution

### On Windows (MinGW/GCC)

```bash
# Exercise 1: Stride Impact
gcc -O0 -o stride_O0.exe exercise1.c
.\stride_O0.exe > resultsO0.txt

gcc -O2 -o stride_O2.exe exercise1.c
.\stride_O2.exe > resultsO2.txt

# Exercise 2: Matrix Multiplication (loop order)
gcc -O2 -o mxm.exe exercise2.c
.\mxm.exe

# Exercise 3: Block Matrix Multiplication
gcc -O2 -o mxm_bloc.exe exercise3_mxm_bloc.c
.\mxm_bloc.exe

# Exercise 4: Memory Debugging
gcc -g -o memory_debug.exe exercise4_memory_debug.c
.\memory_debug.exe
# Note: Use Dr. Memory on Windows instead of Valgrind:
# drmemory -- .\memory_debug.exe

# Exercise 5: HPL Benchmark
gcc -O2 -o hpl_bench.exe exercise5_hpl_benchmark.c
.\hpl_bench.exe
```

### On Linux

```bash
# Exercise 1
gcc -O0 -o stride_O0 exercise1.c && ./stride_O0 > resultsO0.txt
gcc -O2 -o stride_O2 exercise1.c && ./stride_O2 > resultsO2.txt

# Exercise 2
gcc -O2 -o mxm exercise2.c && ./mxm

# Exercise 3
gcc -O2 -o mxm_bloc exercise3_mxm_bloc.c && ./mxm_bloc

# Exercise 4 (with Valgrind)
gcc -g -o memory_debug exercise4_memory_debug.c
valgrind --leak-check=full --track-origins=yes ./memory_debug

# Exercise 5
gcc -O2 -o hpl_bench exercise5_hpl_benchmark.c && ./hpl_bench
```

---

## 📊 Expected Results

### Exercise 1: Stride Impact
- **Stride 1**: Best bandwidth (~3000+ MB/s)
- **Larger strides**: Decreasing bandwidth due to cache misses

### Exercise 2: Loop Order
| Loop Order | Cache Behavior | Expected Speedup |
|------------|---------------|------------------|
| i-j-k | Bad (column access) | 1.0x (baseline) |
| i-k-j | Good (row access) | 2-4x faster |

### Exercise 3: Block Size
| Block Size | Expected Performance |
|------------|---------------------|
| 8-16 | Medium (overhead) |
| 32-64 | **Best** (fits L1 cache) |
| 128-256 | Reduced (cache thrashing) |

---

## 💡 Key Concepts

### 1. Cache Lines
- CPU loads 64 bytes at a time
- Sequential access: use all 64 bytes
- Random access: waste 56 bytes per load

### 2. Row-Major Order
```c
// In C, a[i][j] is stored as: a[0][0], a[0][1], ..., a[1][0], a[1][1], ...
// Access row-wise (varying j) is fast
// Access column-wise (varying i) is slow
```

### 3. Blocking
- Divide matrices into B×B blocks
- Process blocks to maximize cache reuse
- Optimal B: fits 3 blocks in L1 cache

### 4. Performance Metrics
- **Bandwidth (MB/s)**: Data transferred per second
- **GFLOPS**: Billions of floating-point operations per second
- **Efficiency**: Actual / Theoretical peak × 100%

---

## 📈 Analysis Notebook

Open `TP1_Analysis.ipynb` in Jupyter or VS Code to:
1. Visualize results with interactive plots
2. Compare -O0 vs -O2 optimization
3. Analyze optimal block sizes
4. Understand HPL benchmark results

---

## 🔬 Understanding the Results

### Why is optimized code faster?

1. **Better cache utilization**: Sequential access = fewer cache misses
2. **Prefetching**: CPU can predict and preload sequential data
3. **SIMD friendly**: Vector instructions work better on contiguous data

### Why is real performance < theoretical peak?

| Factor | Impact |
|--------|--------|
| Memory latency | ~100 cycles to fetch from RAM |
| Cache misses | Stalls the pipeline |
| Branch mispredictions | Flushes pipeline |
| Instruction dependencies | Limits parallelism |

---

## 📝 Answers to Exercise Questions

### Exercise 2: Why is i-k-j faster?
The i-k-j loop order accesses matrix B row-wise instead of column-wise. In C's row-major storage, this means sequential memory access, which:
- Utilizes full cache lines (64 bytes = 8 doubles)
- Enables CPU prefetching
- Avoids cache misses

### Exercise 3: Why is B=32-64 optimal?
- 3 blocks × B² × 8 bytes ≤ L1 cache (32KB)
- B ≤ √(32KB / 24) ≈ 36
- B=32 or 64 fits perfectly in L1 cache

### Exercise 5: HPL Performance Analysis
- Larger N → better efficiency (overhead amortization)
- Optimal NB = 32-128 (cache fit)
- Real performance: 10-80% of theoretical peak is normal

---

## 👨‍🏫 Author
Student - Abdeljalil Otman 
Date: January 2026
