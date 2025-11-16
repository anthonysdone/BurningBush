// ========================================================================
// Elementwise Operations
// ========================================================================

Tensor *BushAdd(Tensor *a, Tensor *b) {
    BushAssert(BushShapeEquals(a->shape, b->shape, a->ndim), "BushAdd: Shapes must match.");
    Tensor *out = BushAllocTensor(a->shape, a->ndim);
    out->data = BushMalloc(sizeof(F64) * out->size);

    for (I64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }

    return out;
}

// Todo: Fold broadcasting into BushAdd
U0 *BushBroadcastAddInPlace(Tensor *a, Tensor *b) {
    BushAssert(a->ndim == 2 && b->ndim == 1, "BushBroadcastAdd: Unsupported dimensions.");
    BushAssert(a->shape[1] == b->shape[0], "BushBroadcastAdd: Shapes must match.");
    for (I64 i = 0; i < a->shape[0]; i++) {
        for (I64 j = 0; j < a->shape[1]; j++) {
            a->data[i * a->shape[1] + j] += b->data[j];
        }
    }
}

Tensor *BushSub(Tensor *a, Tensor *b) {
    BushAssert(BushShapeEquals(a->shape, b->shape, a->ndim), "BushSub: Shapes must match.");
    Tensor *out = BushAllocTensor(a->shape, a->ndim);
    out->data = BushMalloc(sizeof(F64) * out->size);

    for (I64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }

    return out;
}

Tensor *BushMul(Tensor *a, Tensor *b) {
    BushAssert(BushShapeEquals(a->shape, b->shape, a->ndim), "BushMul: Shapes must match.");
    Tensor *out = BushAllocTensor(a->shape, a->ndim);
    out->data = BushMalloc(sizeof(F64) * out->size);

    for (I64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }

    return out;
}

Tensor *BushDiv(Tensor *a, Tensor *b) {
    BushAssert(BushShapeEquals(a->shape, b->shape, a->ndim), "BushDiv: Shapes must match.");
    Tensor *out = BushAllocTensor(a->shape, a->ndim);
    out->data = BushMalloc(sizeof(F64) * out->size);

    for (I64 i = 0; i < out->size; i++) {
        out->data[i] = a->data[i] / b->data[i];
    }

    return out;
}

// ========================================================================
// Linear Algebra
// ========================================================================

// Todo: Support batched matmul, dot, mv/vm mul. 
Tensor *BushMatMul(Tensor *a, Tensor *b) {
    BushAssert(a->ndim == 2 && b->ndim == 2, "BushMatMul: Only 2D matrices supported.");
    BushAssert(a->shape[1] == b->shape[0], "BushMatMul: Incompatible matrix shapes.");

    I64 m = a->shape[0];
    I64 n = b->shape[1];
    I64 k = a->shape[1];

    I64 out_shape[2] = {m, n};
    Tensor *out = BushZeros(out_shape, 2);

    for (I64 i = 0; i < m; i++) {
        for (I64 j = 0; j < n; j++) {
            F64 sum = 0.0;
            for (I64 p = 0; p < k; p++) {
                sum += a->data[i * k + p] * b->data[p * n + j];
            }
            out->data[i * n + j] = sum;
        }
    }
    return out;
}

Tensor *BushT(Tensor *t) {
    BushAssert(t->ndim == 2, "BushT: Only 2D matrices supported.");
    
    I64 out_shape[2] = {t->shape[1], t->shape[0]};
    Tensor *out = BushEmpty(out_shape, 2);

    for (I64 i = 0; i < t->shape[0]; i++) {
        for (I64 j = 0; j < t->shape[1]; j++) {
            out->data[j * t->shape[0] + i] = t->data[i * t->shape[1] + j];
        }
    }
    return out;
}

// ========================================================================
// Activation Functions
// ========================================================================

Tensor *BushReluTensor(Tensor *t) {
    Tensor *out = BushEmpty(t->shape, t->ndim);

    for (I64 i = 0; i < t->size; i++) {
        out->data[i] = BushRelu(t->data[i]);
    }
    return out;
}

Tensor *BushSigmoidTensor(Tensor *t) {
    Tensor *out = BushEmpty(t->shape, t->ndim);

    for (I64 i = 0; i < t->size; i++) {
        out->data[i] = BushSigmoid(t->data[i]);
    }
    return out;
}

Tensor *BushTanhTensor(Tensor *t) {
    Tensor *out = BushEmpty(t->shape, t->ndim);

    for (I64 i = 0; i < t->size; i++) {
        out->data[i] = BushTanh(t->data[i]);
    }
    return out;
}

// ========================================================================
// Reduction Functions
// ========================================================================

Tensor *BushSumAll(Tensor *t) {
    I64 out_shape[1] = {1};
    Tensor *out = BushEmpty(out_shape, 1);

    F64 sum = 0.0;
    for (I64 i = 0; i < t->size; i++) {
        sum += t->data[i];
    }
    out->data[0] = sum;
    return out;
}

Tensor *BushmeanAll(Tensor *t) {
    Tensor *sum = BushSumAll(t);
    F64 tsize = t->size;
    sum->data[0] /= tsize;
    return sum;
}