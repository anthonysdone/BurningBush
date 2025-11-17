// ===================================================
// Tensor Class
// ===================================================

class Tensor {
    // Data
    F64 *data;
    I64 *shape;
    I64 ndim;
    I64 size;

    // Autograd
    F64 *grad;
    Tensor **parents;
    BackwardFn *backward_fns;
    I64 n_parents;


    // Memory management
    U32 refcount;
    Bool requires_grad;
    Bool is_leaf;
};

class BackwardFn {
    U0 (*fn)(*backward)(BackwardCtx *context);
    BackwardCtx *context; 
};

class BackwardCtx {
    Tensor **saved; 
    I64 n_saved; 
    U0 *extra; 
};

// ===================================================
// Creation Functions
// ===================================================

Tensor *BushTensor(I64 *shape, I64 ndim); 
Tensor *BushFilled(F64 value, I64 *shape, I64 ndim);
Tensor *BushZeros(I64 *shape, I64 ndim);
Tensor *BushOnes(I64 *shape, I64 ndim);
Tensor *BushRandn(I64 *shape, I64 ndim);
Tensor *BushFromArray(F64 *array, I64 *shape, I64 ndim);

// ===================================================
// Memory Management
// ===================================================

Tensor *BushRetain(Tensor *t);
U0 BushRelease(Tensor *t);
U0 BushFreeTensor(Tensor *t);

// ===================================================
// Autograd
// ===================================================

U0 BushBackward(Tensor *t);
U0 BushZeroGrad(Tensor *t);
U0 BushAccumulateGrad(Tensor *t, Tensor *grad);
U0 BushTopoSort(Tensor *t, Tensor ***sorted, I64 *num_tensors);

// ===================================================
// Utilities
// ===================================================

U0 BushPrintTensor(Tensor *t); 
U0 BushAssertShape(Tensor *t, I64 *shape, I64 ndim);