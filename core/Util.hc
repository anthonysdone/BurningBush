#define BUSH_ALIGN 64

#define TRUE 1
#define FALSE 0

public extern "c" F64 exp(F64 f1);
public extern "c" F64 tanh(F64 f1);


// ========================================================================
//  Memory Allocation
// ========================================================================

U0 *BushMalloc(I64 size) {
    U0 *ptr = MAlloc(size); 
    if (!ptr) {
        Print("BushMalloc: Out of memory allocating %lld bytes\n", size);
        throw; 
    }
    return ptr; 
}

U0 *BushCalloc(I64 count, I64 size) {
    U0 *ptr = CAlloc(count, size); 
    if (!ptr) {
        Print("BushCalloc: Out of memory allocating %lld bytes\n", count * size);
        throw; 
    }
    return ptr; 
}

U0 BushFree(U0 *ptr) {
    if (ptr) Free(ptr); 
}

// ========================================================================
//  Shape Operations
// ========================================================================

I64 BushShapeSize(I64 *shape, I64 ndim) {
    I64 size = 1; 
    for (I64 i = 0; i < ndim; i++) {
        size *= shape[i]; 
    }
    return size;
}

I64 *BushShapeCopy(I64 *shape, I64 ndim) {
    I64 *copy = BushMalloc(sizeof(I64) * ndim); 
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
                Print("Shape mismatch at dimension %lld: %lld != %lld\n", i, shape1[i], shape2[i]);
            }
            return FALSE; 
        }
    }
    return TRUE; 
}

Bool BushShapeMatMulCheck(I64 *shape1, I64 *shape2, I64 ndim1, I64 ndim2, Bool verbose=TRUE) {
    if (ndim1 < 2 || ndim2 < 2) {
        if (verbose) {
            Print("MatMul requires both tensors to have at least 2 dimensions\n");
        }
        return FALSE; 
    }
    if (shape1[ndim1 - 1] != shape2[ndim2 - 2]) {
        if (verbose) {
            Print("MatMul shape mismatch: %lld (dim %lld) != %lld (dim %lld)\n", 
                  shape1[ndim1 - 1], ndim1 - 1, shape2[ndim2 - 2], ndim2 - 2);
        }
        return FALSE; 
    }
    return TRUE; 
}

I64 *BushComputeStrides(I64 *shape, I64 ndim) {
    I64 *strides = BushMalloc(sizeof(I64) * ndim); 
    strides[ndim - 1] = 1; 
    for (I64 i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1]; 
    }
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

U0 BushMultiToFlat(I64 *multi_idx, I64 *shape, I64 ndim) {
    I64 flat_idx = 0;
    I64 multiplier = 1;
    for (I64 i = ndim - 1; i >= 0; i--) {
        flat_idx += multi_idx[i] * multiplier;
        multiplier *= shape[i];
    }
    return flat_idx;
}

// ========================================================================
//  Random Number Generation
// ========================================================================

F64 BushRandUniform(F64 a, F64 b) {
    return a + (b - a) * RandF64(); 
}

F64 BushRandNormal(F64 mean, F64 std) {
    F64 u1 = RandF64();
    F64 u2 = RandF64();
    F64 z0 = Sqrt(-2.0 * Log(u1)) * Cos(2.0 * PI * u2);
    return mean + std * z0;
}

U0 BushRandFillUniform(F32 *data, I64 size, F64 a, F64 b) {
    for (I64 i = 0; i < size; i++) {
        data[i] = BushRandUniform(a, b); 
    }
}

U0 BushRandFillNormal(F32 *data, I64 size, F64 mean, F64 std) {
    for (I64 i = 0; i < size; i++) {
        data[i] = BushRandNormal(mean, std); 
    }
}
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
        I64 j = (I64)BushRandUniform(0, (F64)(i + 1));
        I64 temp = perm[i]; 
        perm[i] = perm[j]; 
        perm[j] = temp; 
    }
    return perm;
}

// ========================================================================
//  Math Utilities
// ========================================================================

F32 BushSigmoid(F64 x) {
    return 1.0f / (1.0f + exp(-x));
}

F32 BushRelu(F64 x) {
    return x > 0.0f ? x : 0.0f;
}

F32 BushTanh(F64 x) {
    return tanh(x); 
}

// ========================================================================
//  Debugging
// ========================================================================

U0 BushAssert(Bool condition, I8 *message) {
    if (!condition) {
        Print("Assertion failed: %s\n", message);
        throw; 
    }
}

U0 BushPrintShape(I64 *shape, I64 ndim) {
    Print("(");
    for (I64 i = 0; i < ndim; i++) {
        Print("%lld", shape[i]);
        if (i < ndim - 1) {
            Print(", ");
        }
    }
    Print(")\n");
}