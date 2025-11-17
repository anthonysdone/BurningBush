// ===================================================
// Optimizer Class
// ===================================================

class Optimizer {
    Tensor **params; 
    I64 n_params;
    F64 lr;

    U0 (*step)(Optimizer *self);

    U0 *state; 
}

// ===================================================
// Optimizers
// ===================================================

// Needs work
Optimizer *BushSGD(Tensor **params, I64 n_params, F64 lr=0.01); 
Optimizer *BushSGDStep(Optimizer *self);

Optimizer *BushAdam(Tensor **params, I64 n_params); 
Optimizer *BushAdamStep(Optimizer *self);