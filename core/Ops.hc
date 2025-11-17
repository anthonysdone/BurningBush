// ===================================================
// Elementwise Operations
// ===================================================

Tensor *BushAdd(Tensor *a, Tensor *b);
Tensor *BushBackwardAdd(Tensor *self); 

Tensor *BushSub(Tensor *a, Tensor *b);
Tensor *BushBackwardSub(Tensor *self);

Tensor *BushMul(Tensor *a, Tensor *b);
Tensor *BushBackwardMul(Tensor *self);

Tensor *BushDiv(Tensor *a, Tensor *b);
Tensor *BushBackwardDiv(Tensor *self);

Tensor *BushNeg(Tensor *a);
Tensor *BushBackwardNeg(Tensor *self);

Tensor *BushExp(Tensor *a);
Tensor *BushBackwardExp(Tensor *self);

Tensor *BushLog(Tensor *a);
Tensor *BushBackwardLog(Tensor *self);

Tensor *BushSqrt(Tensor *a);
Tensor *BushBackwardSqrt(Tensor *self);

Tensor *BushPow(Tensor *a, F64 exponent);
Tensor *BushBackwardPow(Tensor *self, F64 exponent);

// ===================================================
// Reductions
// ===================================================

Tensor *BushSum(Tensor *a, I64 dim=-1);
Tensor *BushBackwardSum(Tensor *self, I64 dim);

Tensor *BushMean(Tensor *a, I64 dim=-1);
Tensor *BushBackwardMean(Tensor *self, I64 dim);

Tensor *BushMax(Tensor *a, I64 dim=-1);
Tensor *BushBackwardMax(Tensor *self, I64 dim);

// ===================================================
// Linear Algebra
// ===================================================

Tensor *BushDot(Tensor *a, Tensor *b);
Tensor *BushBackwardDot(Tensor *self);

Tensor *BushMVMul(Tensor *a, Tensor *b);
Tensor *BushBackwardMVMul(Tensor *self);

Tensor *BushVMMul(Tensor *a, Tensor *b);
Tensor *BushBackwardVMMul(Tensor *self);

Tensor *BushMMMul(Tensor *a, Tensor *b);
Tensor *BushBackwardMMMul(Tensor *self);

Tensor *BushMatMul(Tensor *a, Tensor *b);
Tensor *BushBackwardMatMul(Tensor *self);

Tensor *BushTranspose(Tensor *a);
Tensor *BushBackwardTranspose(Tensor *self);

// ===================================================
// Convolution Operations
// ===================================================

Tensor *BushConv2D(Tensor *a, Tensor *weight, I64 stride, I64 padding);
Tensor *BushBackwardConv2D(Tensor *self);

Tensor *BushMaxPool2D(Tensor *a, I64 kernel_size, I64 stride, I64 padding);
Tensor *BushBackwardMaxPool2D(Tensor *self);

// ===================================================
// Activations
// ===================================================

Tensor *BushRelu(Tensor *a);
Tensor *BushBackwardRelu(Tensor *self);

Tensor *BushLeakyRelu(Tensor *a, F64 alpha);
Tensor *BushBackwardLeakyRelu(Tensor *self);

Tensor *BushSigmoid(Tensor *a);
Tensor *BushBackwardSigmoid(Tensor *self);

Tensor *BushTanh(Tensor *a);
Tensor *BushBackwardTanh(Tensor *self);

Tensor *BushSoftmax(Tensor *a, I64 dim=-1);
Tensor *BushBackwardSoftmax(Tensor *self);

// ===================================================
// Loss Functions
// ===================================================

Tensor *BushMSE(Tensor *pred, Tensor *target);
Tensor *BushBackwardMSE(Tensor *self);

Tensor *BushCrossEntropy(Tensor *pred, Tensor *target);
Tensor *BushBackwardCrossEntropy(Tensor *self);

Tensor *BushBCE(Tensor *pred, Tensor *target);
Tensor *BushBackwardBCE(Tensor *self);

// ===================================================
// Views
// ===================================================

Tensor *BushReshape(Tensor *a, I64 *new_shape, I64 new_ndim);
Tensor *BushBackwardReshape(Tensor *self);

Tensor *BushFlatten(Tensor *a);
Tensor *BushBackwardFlatten(Tensor *self);