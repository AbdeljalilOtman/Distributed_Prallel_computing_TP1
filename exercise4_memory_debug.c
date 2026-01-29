

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 5

/* -----------------------------------------------------------------------------
 * ALLOCATE ARRAY
 * 
 * Creates a new array of integers on the heap.
 * 
 * IMPORTANT: Every malloc() must have a matching free()!
 * The caller is responsible for freeing this memory.
 * -------------------------------------------------------------------------- */
int* allocate_array(int size) 
{
    /* malloc returns a pointer to newly allocated memory */
    int *arr = (int*)malloc(size * sizeof(int));
    
    /* Always check if allocation succeeded! */
    if (!arr) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    printf("  [DEBUG] Allocated %zu bytes at address %p\n", 
           size * sizeof(int), (void*)arr);
    
    return arr;
}

/* -----------------------------------------------------------------------------
 * INITIALIZE ARRAY
 * 
 * Fills the array with some values.
 * -------------------------------------------------------------------------- */
void initialize_array(int *arr, int size) 
{
    if (!arr) return;  /* Safety check for NULL pointer */
    
    for (int i = 0; i < size; i++) 
    {
        arr[i] = i * 10;  /* Values: 0, 10, 20, 30, 40 */
    }
    printf("  [DEBUG] Initialized array with values\n");
}

/* -----------------------------------------------------------------------------
 * PRINT ARRAY
 * 
 * Displays the contents of an array.
 * -------------------------------------------------------------------------- */
void print_array(int *arr, int size) 
{
    if (!arr) return;
    
    printf("  Array elements: ");
    for (int i = 0; i < size; i++) 
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

/* -----------------------------------------------------------------------------
 * DUPLICATE ARRAY
 * 
 * Creates a copy of an array.
 * 
 * NOTE: This function allocates NEW memory!
 * The caller must free both the original AND the copy.
 * -------------------------------------------------------------------------- */
int* duplicate_array(int *arr, int size) 
{
    if (!arr) return NULL;
    
    /* Allocate memory for the copy */
    int *copy = (int*)malloc(size * sizeof(int));
    
    if (!copy) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    /* Copy the data using memcpy (faster than a loop) */
    memcpy(copy, arr, size * sizeof(int));
    
    printf("  [DEBUG] Created copy at address %p\n", (void*)copy);
    
    return copy;
}

/* -----------------------------------------------------------------------------
 * FREE MEMORY
 * 
 * FIXED VERSION: This function properly frees the allocated memory.
 * 
 * Original bug: This function was empty, causing a memory leak!
 * 
 * BEST PRACTICES:
 * 1. Always check for NULL before freeing
 * 2. Set pointer to NULL after freeing (prevents double-free bugs)
 * 3. Document who is responsible for freeing memory
 * -------------------------------------------------------------------------- */
void free_memory(int *arr) 
{
    if (arr != NULL)  /* Safety check */
    {
        printf("  [DEBUG] Freeing memory at address %p\n", (void*)arr);
        free(arr);
        /* Note: We can't set arr = NULL here because arr is a local copy
         * of the pointer. The caller should set their pointer to NULL. */
    }
}

/* -----------------------------------------------------------------------------
 * MAIN PROGRAM - DEMONSTRATING MEMORY MANAGEMENT
 * -------------------------------------------------------------------------- */
int main() 
{
    printf("====================================================================\n");
    printf("       Exercise 4: Memory Management and Debugging                  \n");
    printf("====================================================================\n\n");
    
    printf("STEP 1: Allocate original array\n");
    printf("---------------------------------\n");
    int *array = allocate_array(SIZE);
    
    printf("\nSTEP 2: Initialize array\n");
    printf("-------------------------\n");
    initialize_array(array, SIZE);
    
    printf("\nSTEP 3: Print original array\n");
    printf("-----------------------------\n");
    print_array(array, SIZE);
    
    printf("\nSTEP 4: Create a duplicate\n");
    printf("---------------------------\n");
    int *array_copy = duplicate_array(array, SIZE);
    
    printf("\nSTEP 5: Print the copy\n");
    printf("-----------------------\n");
    print_array(array_copy, SIZE);
    
    printf("\nSTEP 6: Free memory (IMPORTANT!)\n");
    printf("---------------------------------\n");
    
    /* 
     * FIXED: We now free BOTH arrays!
     * 
     * Original code only freed 'array', not 'array_copy'.
     * This caused a memory leak of SIZE * sizeof(int) = 20 bytes.
     */
    free_memory(array);
    array = NULL;  /* Good practice: set to NULL after free */
    
    free_memory(array_copy);  /* This was missing - THE BUG! */
    array_copy = NULL;
    
    printf("\n");
    printf("====================================================================\n");
    printf("                         EXPLANATION                                \n");
    printf("====================================================================\n");
    printf("\n");
    printf("MEMORY LEAK FIXED!\n");
    printf("------------------\n");
    printf("The original code had a memory leak:\n");
    printf("  * allocate_array() allocated memory for 'array'\n");
    printf("  * duplicate_array() allocated memory for 'array_copy'\n");
    printf("  * Only 'array' was freed, 'array_copy' was leaked!\n");
    printf("\n");
    printf("HOW TO DETECT WITH VALGRIND:\n");
    printf("----------------------------\n");
    printf("1. Compile: gcc -g -o memory_debug exercise4_memory_debug.c\n");
    printf("2. Run: valgrind --leak-check=full ./memory_debug\n");
    printf("\n");
    printf("With the bug, Valgrind would show:\n");
    printf("  LEAK SUMMARY:\n");
    printf("    definitely lost: 20 bytes in 1 blocks\n");
    printf("\n");
    printf("After fixing, Valgrind shows:\n");
    printf("  All heap blocks were freed -- no leaks are possible\n");
    printf("\n");
    printf("ON WINDOWS (without Valgrind):\n");
    printf("------------------------------\n");
    printf("Use Address Sanitizer:\n");
    printf("  gcc -fsanitize=address -g -o memory_debug exercise4_memory_debug.c\n");
    printf("  ./memory_debug\n");
    printf("\n");
    printf("Or use Dr. Memory:\n");
    printf("  drmemory -- memory_debug.exe\n");
    printf("\n");
    
    return 0;
}

/*
 * =============================================================================
 * COMMON MEMORY BUGS AND HOW TO AVOID THEM
 * =============================================================================
 * 
 * 1. MEMORY LEAK (what we fixed above)
 *    - Problem: malloc without free
 *    - Fix: Always pair malloc with free
 *    - Tip: Use RAII pattern or smart pointers in C++
 * 
 * 2. DOUBLE FREE
 *    - Problem: Calling free() twice on the same pointer
 *    - Fix: Set pointer to NULL after freeing
 *    - Example:
 *        free(ptr);
 *        ptr = NULL;  // Now free(ptr) is safe (free(NULL) is a no-op)
 * 
 * 3. USE AFTER FREE
 *    - Problem: Accessing memory after freeing it
 *    - Fix: Set pointer to NULL and check before use
 * 
 * 4. BUFFER OVERFLOW
 *    - Problem: Writing past the end of an array
 *    - Fix: Always check array bounds
 *    - Example: for (i = 0; i < SIZE; i++)  // not i <= SIZE
 * 
 * 5. UNINITIALIZED MEMORY
 *    - Problem: Using malloc'd memory without initializing
 *    - Fix: Use calloc() or memset() to zero memory
 *    - Example: int *arr = calloc(n, sizeof(int));  // zeros memory
 * 
 * =============================================================================
 */
