class Module; 

// Module type tags
#define MODULE_LINEAR 1
#define MODULE_SEQUENTIAL 2
#define MODULE_RELU 3
#define MODULE_SIGMOID 4
#define MODULE_TANH 5

class Module {
    Tensor **params; 
    I64 num_params; 

    I64 module_type;  // Use type tag instead of function pointer

    U0 *extra; 
    U8 *name; 
    Bool training; 
};

class LinearExtra {
    I64 in_features; 
    I64 out_features; 
    Bool has_bias; 
};

class Sequential {
    Module base; 
    Module **modules; 
    I64 num_modules; 
};

// ========================================================================
// Forward Declarations
// ========================================================================

Tensor *LinearForward(Module *self, Tensor *x);
Tensor *SequentialForward(Module *self, Tensor *x);
Tensor *ReluForward(Module *self, Tensor *x);
Tensor *SigmoidForward(Module *self, Tensor *x);
Tensor *TanhForward(Module *self, Tensor *x);
Tensor *BushForward(Module *m, Tensor *input);

// ========================================================================
// Linear Layer
// ======================================================================== 

Module *BushLinear(I64 in_features, I64 out_features, Bool bias) {
    Module *m = BushCalloc(1, sizeof(Module));

    if (bias) m->num_params = 2;
    else m->num_params = 1;
    m->params = BushMalloc(sizeof(Tensor *) * m->num_params);

    I64 weight_shape[2] = {out_features, in_features};
    m->params[0] = BushRandn(weight_shape, 2);

    // Skip weight scaling for now
    // F64 std = sqrt(2.0 / (in_features + out_features));
    // for (I64 i = 0; i < m->params[0]->size; i++) {
    //     m->params[0]->data[i] *= std; 
    // }

    if (bias) {
        I64 bias_shape[1] = {out_features};
        m->params[1] = BushZeros(bias_shape, 1);
    }

    m->module_type = MODULE_LINEAR;

    LinearExtra *extra = BushMalloc(sizeof(LinearExtra));
    extra->in_features = in_features;
    extra->out_features = out_features;
    extra->has_bias = bias;
    m->extra = extra;

    m->name = "Linear";
    m->training = TRUE;

    return m; 
}

Tensor *LinearForward(Module *self, Tensor *x) {
    Tensor *W = self->params[0];
    Tensor *W_T = BushT(W); 
    Tensor *out = BushMatMul(x, W_T);
    BushRelease(W_T); 

    LinearExtra *extra = self->extra;
    if (extra->has_bias) {
        Tensor *b = self->params[1]; 
        BushBroadcastAddInPlace(out, b);
    }
    return out;
}

// ========================================================================
// Sequential Container
// ========================================================================

Module *BushSequential(Module **modules, I64 num_modules) {
    Module *m = BushCalloc(1, sizeof(Module));

    m->num_params = 0; 
    m->params = 0;

    Sequential *seq = BushMalloc(sizeof(Sequential));
    seq->modules = modules;
    seq->num_modules = num_modules;

    m->extra = seq; 
    m->module_type = MODULE_SEQUENTIAL;
    m->name = "Sequential";
    m->training = TRUE;

    return m;
}

U0 *SequentialAdd(Sequential *seq, Module *module) {
    seq->modules = BushRealloc(seq->modules, sizeof(Module *) * (seq->num_modules + 1));
    seq->modules[seq->num_modules] = module; 
    seq->num_modules += 1; 
}

Tensor *SequentialForward(Module *self, Tensor *x);  // Forward declaration

Tensor *SequentialForward(Module *self, Tensor *x) {
    Sequential *seq;
    seq = self->extra; 
    Tensor *out = BushRetain(x); 

    for (I64 i = 0; i < seq->num_modules; i++) {
        Tensor *prev = out; 
        Module *mod = seq->modules[i];
        out = BushForward(mod, prev);  // Use BushForward instead
        BushRelease(prev);
    }
    return out;
}

// ========================================================================
// Activation Modules
// ========================================================================

Tensor *ReluForward(Module *self, Tensor *x) { 
    return BushReluTensor(x); 
}

Module *BushReluModule() {
    Module *m = BushCalloc(1, sizeof(Module));
    m->num_params = 0; 
    m->params = 0;
    m->module_type = MODULE_RELU;
    m->name = "ReLU";
    m->training = TRUE;
    return m;
}

Tensor *SigmoidForward(Module *self, Tensor *x) { 
    return BushSigmoidTensor(x); 
}

Module *BushSigmoidModule() {
    Module *m = BushCalloc(1, sizeof(Module));
    m->num_params = 0; 
    m->params = 0;
    m->module_type = MODULE_SIGMOID;
    m->name = "Sigmoid";
    m->training = TRUE;
    return m;
}

Tensor *TanhForward(Module *self, Tensor *x) { 
    return BushTanhTensor(x); 
}

Module *BushTanhModule() {
    Module *m = BushCalloc(1, sizeof(Module));
    m->num_params = 0; 
    m->params = 0;
    m->module_type = MODULE_TANH;
    m->name = "Tanh";
    m->training = TRUE;
    return m;
}

// ========================================================================
// Parameter Management
// ========================================================================

U0 BushGetParametersRecursive(Module *m, Tensor ***all_params, I64 *count) {
    for (I64 i = 0; i < m->num_params; i++) {
        (*all_params)[*count] = m->params[i]; 
        (*count)++;
    }

    if (m->module_type == MODULE_SEQUENTIAL) {
        Sequential *seq;
        seq = m->extra; 
        for (I64 i = 0; i < seq->num_modules; i++) {
            BushGetParametersRecursive(seq->modules[i], all_params, count);
        }
    }
}

I64 BushCountParameters(Module *m) {
    I64 count = m->num_params; 

    if (m->module_type == MODULE_SEQUENTIAL) {
        Sequential *seq;
        seq = m->extra; 
        for (I64 i = 0; i < seq->num_modules; i++) {
            count += BushCountParameters(seq->modules[i]);
        }
    }

    return count; 
}

Tensor **BushParameters(Module *m, I64 *num_params) {
    *num_params = BushCountParameters(m);
    Tensor **params = BushMalloc(sizeof(Tensor *) * (*num_params));
    I64 count = 0;
    BushGetParametersRecursive(m, &params, &count);
    return params;
}

I64 BushNumParameters(Module *m) {
    I64 num_params = 0; 
    Tensor **params = BushParameters(m, &num_params);
    BushFree(params); 
    return num_params; 
}

// ========================================================================
// Forward Pass Dispatcher
// ========================================================================

Tensor *BushForward(Module *m, Tensor *input) {
    if (m->module_type == MODULE_LINEAR) {
        return LinearForward(m, input);
    }
    else if (m->module_type == MODULE_SEQUENTIAL) {
        return SequentialForward(m, input);
    }
    else if (m->module_type == MODULE_RELU) {
        return ReluForward(m, input);
    }
    else if (m->module_type == MODULE_SIGMOID) {
        return SigmoidForward(m, input);
    }
    else if (m->module_type == MODULE_TANH) {
        return TanhForward(m, input);
    }
    
    "ERROR: Unknown module type %d\n", m->module_type;
    return 0;
}

Tensor *BushCall(Module *m, Tensor *input) {
    return BushForward(m, input);
}