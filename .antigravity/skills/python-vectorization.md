# Python Vectorization & Performance Protocol

You are an expert performance optimization agent. When reviewing or writing Python code for this project, you must adhere strictly to the following constraints to meet the < 15 ms pipeline latency target:

1. **NO PYTHON LOOPS:** Ruthlessly eliminate `for` and `while` loops over image arrays or coordinate lists.
2. **NUMPY/PYTORCH NATIVE:** Replace iterations with native vectorized operations (e.g., `np.where`, `np.einsum`, boolean indexing).
3. **MEMORY ALLOCATION:** Pre-allocate arrays where possible. Avoid dynamically appending to lists inside hot paths.
4. **PARALLELIZATION:** When processing multiple columns for sub-pixel extraction, process them as a 2D matrix batch rather than iterating column-by-column.

If you detect unoptimized loops, you must rewrite them using matrix math and explain the Big-O time complexity difference.