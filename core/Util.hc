#define BUSH_ALIGN 64

I64 malloc_count = 0;

public extern "c" F64 exp(F64 f1);
public extern "c" F64 tanh(F64 f1);

public extern "c" F64 log(F64 f1);
public extern "c" F64 sqrt(F64 f1);
public extern "c" F64 cos(F64 f1);

extern "c" I32 strcmp(U8 *__s1, U8 *__s2);
extern "c" I32 rand();
#define RAND_MAX 0x7FFF
#define PI 3.14159265358979323846

// ========================================================================
// Memory Allocation
// ========================================================================

U0 *BushMalloc(I64 size) {
    malloc_count++;
    U0 *ptr = MAlloc(size); 
    if (!ptr) {
        "BushMalloc: Out of memory allocating %lld bytes\n", size;
    }
    return ptr; 
}

U0 *BushCalloc(I64 count, I64 size) {
    U0 *ptr = CAlloc(count, size); 
    if (!ptr) {
        "BushCalloc: Out of memory allocating %lld bytes\n", count * size;
    }
    return ptr; 
}

U0 *BushRealloc(U0 *ptr, I64 new_size) {
    U0 *new_ptr = MAlloc(new_size); 
    if (!new_ptr) {
        "BushRealloc: Out of memory reallocating to %lld bytes\n", new_size;
        return ptr;
    }
    if (ptr) {
        MemCpy(new_ptr, ptr, new_size);
        Free(ptr);
    }
    return new_ptr; 
}

U0 BushFree(U0 *ptr) {
    if (ptr) Free(ptr); 
}

// ========================================================================
// Shape Operations
// ========================================================================

I64 BushShapeSize(I64 *shape, I64 ndim) {
    I64 size = 1; 
    for (I64 i = 0; i < ndim; i++) {
        size *= shape[i]; 
    }
    return size;
}

I64 *BushShapeCopy(I64 *shape, I64 ndim) {
    I64 size_bytes = sizeof(I64) * ndim;
    I64 *copy = BushMalloc(size_bytes);
    if (!copy) {
        "BushShapeCopy: BushMalloc failed for %lld bytes\n", size_bytes;
        return 0;
    }
    I64 i; 
    for (i = 0; i < ndim; i++) {
        copy[i] = shape[i]; 
    }
    return copy; 
}

Bool BushShapeEquals(I64 *shape1, I64 *shape2, I64 ndim, Bool verbose=TRUE) {
    for (I64 i = 0; i < ndim; i++) {
        if (shape1[i] != shape2[i]) {
            if (verbose) {
                "Shape mismatch at dimension %lld: %lld != %lld\n", i, shape1[i], shape2[i];
            }
            return FALSE; 
        }
    }
    return TRUE; 
}

Bool BushShapeMatMulCheck(I64 *shape1, I64 *shape2, I64 ndim1, I64 ndim2, Bool verbose=TRUE) {
    if (ndim1 < 2 || ndim2 < 2) {
        if (verbose) {
            "MatMul requires both tensors to have at least 2 dimensions\n";
        }
        return FALSE; 
    }
    if (shape1[ndim1 - 1] != shape2[ndim2 - 2]) {
        if (verbose) {
            "MatMul shape mismatch: %lld (dim %lld) != %lld (dim %lld)\n", 
                  shape1[ndim1 - 1], ndim1 - 1, shape2[ndim2 - 2], ndim2 - 2;
        }
        return FALSE; 
    }
    return TRUE; 
}

I64 *BushComputeStrides(I64 *shape, I64 ndim) {
    "[ComputeStrides] Start: ndim=%lld\n", ndim;
    I64 *strides = BushMalloc(sizeof(I64) * ndim);
    "[ComputeStrides] Allocated strides array\n";
    strides[ndim - 1] = 1;
    "[ComputeStrides] Set strides[%lld] = 1\n", ndim - 1;
    for (I64 i = ndim - 2; i >= 0; i--) {
        "[ComputeStrides] Loop: i=%lld\n", i;
        strides[i] = strides[i + 1] * shape[i + 1];
        "[ComputeStrides] Set strides[%lld] = %lld\n", i, strides[i];
    }
    "[ComputeStrides] Complete\n";
    return strides;
}

I64 BushComputeOffset(I64 *indices, I64 *strides, I64 ndim) {
    I64 offset = 0; 
    for (I64 i = 0; i < ndim; i++) {
        offset += indices[i] * strides[i]; 
    }
    return offset; 
}

U0 BushFlatToMulti(I64 flat_idx, I64 *shape, I64 ndim, I64 *multi_idx) {
    for (I64 i = ndim - 1; i >= 0; i--) {
        multi_idx[i] = flat_idx % shape[i]; 
        flat_idx /= shape[i]; 
    }
}

I64 BushMultiToFlat(I64 *multi_idx, I64 *shape, I64 ndim) {
    I64 flat_idx = 0;
    I64 multiplier = 1;
    for (I64 i = ndim - 1; i >= 0; i--) {
        flat_idx += multi_idx[i] * multiplier;
        multiplier *= shape[i];
    }
    return flat_idx;
}

// ========================================================================
// Random Number Generation
// ========================================================================

F64 BushRandUniform(F64 a, F64 b) {
    F64 r = rand();
    return a + (b - a) * (r / RAND_MAX); 
}

F64 BushRandNormal(F64 mean, F64 std) {
    F64 r1 = rand();
    F64 r2 = rand();
    F64 u1 = r1 / RAND_MAX;
    F64 u2 = r2 / RAND_MAX;
    F64 z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return mean + std * z0;
}

U0 BushRandFillUniform(F64 *data, I64 size, F64 a, F64 b) {
    for (I64 i = 0; i < size; i++) {
        data[i] = BushRandUniform(a, b); 
    }
}

U0 BushRandFillNormal(F64 *data, I64 size, F64 mean, F64 std) {
    for (I64 i = 0; i < size; i++) {
        data[i] = BushRandNormal(mean, std); 
    }
}

I64 *BushRandPerm(I64 n) {
    I64 *perm = BushMalloc(sizeof(I64) * n); 
    for (I64 i = 0; i < n; i++) {
        perm[i] = i; 
    }

    for (I64 i = n - 1; i > 0; i--) {
        F64 iplus1 = i + 1;
        F64 rval = BushRandUniform(0.0, iplus1);
        I64 j = rval(I64);
        I64 temp = perm[i]; 
        perm[i] = perm[j]; 
        perm[j] = temp; 
    }
    return perm;
}

// ========================================================================
//  Math Utilities
// ========================================================================

F64 BushSigmoid(F64 x) {
    return 1.0 / (1.0 + exp(-x));
}

F64 BushRelu(F64 x) {
    if (x > 0.0) return x;
    return 0.0;
}

F64 BushTanh(F64 x) {
    return tanh(x); 
}

// ========================================================================
// Debugging
// ========================================================================

U0 BushAssert(Bool condition, U8 *message) {
    if (!condition) {
        "Assertion failed: %s\n", message;
    }
}

U0 BushPrintShape(I64 *shape, I64 ndim) {
    "(";
    for (I64 i = 0; i < ndim; i++) {
        "%lld", shape[i];
        if (i < ndim - 1) {
            ", ";
        }
    }
    ")\n";
}