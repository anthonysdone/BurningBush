#include "Util.hc"

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
//  Tensor Creation
// ========================================================================

Tensor *BushAllocTensor(I64 *shape, I64 ndim) {
    Tensor *t = BushCalloc(1, sizeof(Tensor));

    t->ndim = ndim;
    t->shape = BushShapeCopy(shape, ndim); 
    t->size = BushShapeSize(shape, ndim); 
    t->strides = BushComputeStrides(shape, ndim); 
    t->data = NULL; 
    t->refcount = 1; 
    t->is_view = FALSE; 
    t->is_contiguous = TRUE;
    t->base = NULL;
}

Tensor *BushTensor(F32 *data, I64 *shape, I64 ndim) {
    Tensor *t = BushAllocTensor(shape, ndim); 

    t->data = BushMalloc(sizeof(F64) * t->size);
    for (I64 i = 0; i < t->size; i++) {
        t->data[i] = (F64)data[i]; 
    }

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
    t->data = BushMalloc(sizeof(F64) * t->size);
    BushRandFillNormal(t->data, t->size, 0.0, 1.0);
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
    I64 n = (I64)((end - start) / step);
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

    F64 step = (end - start) / (F64)(num - 1);
    for (I64 i = 0; i < num; i++) {
        t->data[i] = start + i * step; 
    }
    return t;
}

// ========================================================================
//  Memory Management
// ========================================================================

Tensor *BushRetain(Tensor *t) {
    if (t) t->refcount++; 
    return t; 
}

U0 BushRelease(Tensor *t) {
    if (!t) return; 

    t->refcount--;
    if (t->refcount == 0) {
        if (t->is_view && t->base) BushRelease(t->data); 
        if (t->base) BushRelease(t->base); 
        BushFree(t->shape);
        BushFree(t->strides);
        BushFree(t); 
    }
}

U0 BushFreeTensor(Tensor *t) {
    BushRelease(t); 
}

// ========================================================================
//  Tensor Properties
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
//  Data Access
// ========================================================================

