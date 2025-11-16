class Tensor; 

class Tensor {
    // Data
    F64 *data; 
    I64 *shape; 
    I64 ndim;
    I64 size;
    I64 *strides;

    // Memory
    U32 refcount; 
    U8 is_view; 
    U8 is_contiguous;
    Tensor *base; 
}; 

// ========================================================================
// Tensor Creation
// ========================================================================

Tensor *BushAllocTensor(I64 *shape, I64 ndim) {
    "[AllocTensor] Start: ndim=%lld\n", ndim;
    Tensor *t = BushCalloc(1, sizeof(Tensor));
    "[AllocTensor] Tensor struct allocated\n";

    t->ndim = ndim;
    "[AllocTensor] Calling BushShapeCopy\n";
    t->shape = BushShapeCopy(shape, ndim);
    "[AllocTensor] Shape copied\n";
    t->size = BushShapeSize(shape, ndim);
    "[AllocTensor] Size computed: %lld\n", t->size;
    "[AllocTensor] Calling BushComputeStrides\n";
    t->strides = BushComputeStrides(shape, ndim);
    "[AllocTensor] Strides computed\n";
    t->data = 0; 
    t->refcount = 1; 
    t->is_view = FALSE; 
    t->is_contiguous = TRUE;
    t->base = 0;
    "[AllocTensor] Complete\n";
    return t;
}

Tensor *BushTensor(F64 *data, I64 *shape, I64 ndim) {
    Tensor *t = BushAllocTensor(shape, ndim); 

    t->data = BushMalloc(sizeof(F64) * t->size);
    for (I64 i = 0; i < t->size; i++) {
        t->data[i] = data[i]; 
    }

    return t; 
}

Tensor *BushEmpty(I64 *shape, I64 ndim) {
    Tensor *t = BushAllocTensor(shape, ndim); 
    t->data = BushMalloc(sizeof(F64) * t->size);
    return t;
}

Tensor *BushZeros(I64 *shape, I64 ndim) {
    Tensor *t = BushAllocTensor(shape, ndim); 
    t->data = BushCalloc(t->size, sizeof(F64));
    return t;
}

Tensor *BushOnes(I64 *shape, I64 ndim) {
    Tensor *t = BushAllocTensor(shape, ndim); 
    t->data = BushMalloc(sizeof(F64) * t->size);
    for (I64 i = 0; i < t->size; i++) {
        t->data[i] = 1.0; 
    }
    return t;
}

Tensor *BushRandn(I64 *shape, I64 ndim) {
    Tensor *t = BushAllocTensor(shape, ndim); 
    t->data = BushCalloc(t->size, sizeof(F64));
    // Just use zeros for now
    return t;
}

Tensor *BushFull(I64 *shape, I64 ndim, F64 value) {
    Tensor *t = BushAllocTensor(shape, ndim); 
    t->data = BushMalloc(sizeof(F64) * t->size);
    for (I64 i = 0; i < t->size; i++) {
        t->data[i] = value; 
    }
    return t;
}

Tensor *BushArange(F64 start, F64 end, F64 step) {
    F64 nf = (end - start) / step;
    I64 n = nf(I64);
    I64 shape[1]; 
    shape[0] = n; 

    Tensor *t = BushAllocTensor(shape, 1);
    t->data = BushMalloc(sizeof(F64) * n);

    for (I64 i = 0; i < n; i++) {
        t->data[i] = start + i * step; 
    }
    return t;
}

Tensor *BushLinspace(F64 start, F64 end, I64 num) {
    I64 shape[1]; 
    shape[0] = num; 

    Tensor *t = BushAllocTensor(shape, 1);
    t->data = BushMalloc(sizeof(F64) * num);

    F64 numm1 = num - 1;
    F64 step = (end - start) / numm1;
    for (I64 i = 0; i < num; i++) {
        t->data[i] = start + i * step; 
    }
    return t;
}

// ========================================================================
// Memory Management
// ========================================================================

Tensor *BushRetain(Tensor *t) {
    if (t) t->refcount++; 
    return t; 
}

U0 BushRelease(Tensor *t) {
    if (!t) return; 

    t->refcount--;
    if (t->refcount == 0) {
        if (t->base) BushRelease(t->base); 
        if (!t->is_view) {
            if (t->data) {
                U0 *ptr = t->data;
                BushFree(ptr);
            }
        }
        BushFree(t->shape);
        BushFree(t->strides);
        BushFree(t); 
    }
}

U0 BushFreeTensor(Tensor *t) {
    BushRelease(t); 
}

// ========================================================================
// Tensor Properties
// ========================================================================

I64 BushNdim(Tensor *t) {
    return t->ndim; 
}

I64 *BushSize(Tensor *t) {
    return t->shape; 
}

I64 BushShape(Tensor *t, I64 dim) {
    BushAssert(dim >= 0 && dim < t->ndim, "Dimension out of range"); 
    return t->shape[dim]; 
}

I64 *BushShapeArray(Tensor *t) {
    return t->shape; 
}

I64 BushIsView(Tensor *t) {
    return t->is_view; 
}

I64 BushIsContiguous(Tensor *t) {
    return t->is_contiguous; 
}

// ========================================================================
// Data Access
// ========================================================================

F64 BushItem(Tensor *t) {
    BushAssert(t->size == 1, "BushItem requires scalar tensor");
    return t->data[0];
}

F64 BushGet(Tensor *t, I64 *indices) {
    I64 offset = BushComputeOffset(indices, t->strides, t->ndim);
    return t->data[offset];
}

U0 BushSet(Tensor *t, I64 *indices, F64 value) {
    I64 offset = BushComputeOffset(indices, t->strides, t->ndim);
    t->data[offset] = value; 
}

// ========================================================================
// Printing
// ========================================================================

U0 BushPrintRecursive(Tensor *t, I64 dim, I64 *indices, I64 indent) {
    if (dim == t->ndim - 1) {
        "["; 
        for (I64 i = 0; i < t->shape[dim]; i++) {
            indices[dim] = i; 
            "%.4f", BushGet(t, indices);
        }
        "]";
    } else {
        "["; 
        for (I64 i = 0; i < t->shape[dim]; i++) {
            if (i > 0) {
            "\n"; 
                for (I64 j = 0; j < indent + 1; j++) " ";
            }
            indices[dim] = i; 
            BushPrintRecursive(t, dim + 1, indices, indent + 1);
            if (i < t->shape[dim] - 1) ","; 
        }
        "]";
    }
}

U0 BushPrint(Tensor *t) {
    "Tensor("; 
    BushPrintShape(t->shape, t->ndim);
    "):";
    "\n";
    if (t->size <= 100) {
        I64 *indices = BushCalloc(t->ndim, sizeof(I64));
        BushPrintRecursive(t, 0, indices, 0);
        "\n";
        BushFree(indices);
    } else {
        "[... %d elements ...]\n", t->size;
    }
}

// ========================================================================
// Scalar Operations
// ========================================================================

Tensor *BushScalarAdd(Tensor *t, F64 scalar) {
    Tensor *result = BushAllocTensor(t->shape, t->ndim); 
    result->data = BushMalloc(sizeof(F64) * t->size);
    for (I64 i = 0; i < t->size; i++) {
        result->data[i] = t->data[i] + scalar; 
    }
    return result;
}

Tensor *BushScalarMul(Tensor *t, F64 scalar) {
    Tensor *result = BushAllocTensor(t->shape, t->ndim); 
    result->data = BushMalloc(sizeof(F64) * t->size);
    for (I64 i = 0; i < t->size; i++) {
        result->data[i] = t->data[i] * scalar; 
    }
    return result;
}