// ===================================================
// Layer Class
// ===================================================

class Layer {
    Tensor **params; 
    I64 n_params; 
    U8 *name;
    U0 *context; 

    Tensor *(*forward)(Layer *self, Tensor *input);
};

// ===================================================
// Context Classes
// ===================================================

class NoContext; 
class LinearContext;
class LeakyReluContext; 
class SoftmaxContext;
class Conv2DContext;
class MaxPool2DContext;
class ReductionContext; 

// ===================================================
// Layer Creation
// ===================================================

// Linear
Layer *BushLinear(I64 in_features, I64 out_features, Bool bias);

// Reductions
Layer *BushSumLayer(I64 axis, Bool keepdim);
Layer *BushMeanLayer(I64 axis, Bool keepdim);
Layer *BushMaxLayer(I64 axis, Bool keepdim);

// Convolution
Layer *BushConv2DLayer(I64 in_channels, I64 out_channels, I64 kernel_size, I64 stride, I64 padding, Bool bias);
Layer *BushMaxPool2DLayer(I64 kernel_size, I64 stride, I64 padding);
Layer *BushFlattenLayer();

// Activations
Layer *BushReLULayer();
Layer *BushLeakyReLULayer(F64 negative_slope);
Layer *BushSigmoidLayer();
Layer *BushTanhLayer();
Layer *BushSoftmaxLayer(I64 axis);

// ===================================================
// Layer Destruction
// ===================================================

U0 BushFreeLayer(Layer *layer);