// ===================================================
// Network Class
// ===================================================

class Network {
    Layer **layers; 
    I64 n_layers; 

    // Cached parameters
    Tensor **params;
    I64 n_params;
}; 

// ===================================================
// Network Management
// ===================================================

Network *BushNetwork(Layer **layers);
U0 *BushAddLayer(Network *net, Layer *layer);
U0 BushFreeNetwork(Network *net);


// ===================================================
// Training and Use
// ===================================================

Tensor *BushForward(Network *net, Tensor *input);
U0 BushTrainStep(Network *net, Tensor *input, Tensor *target, Optimizer *opt, F64 (*loss_fn)(Tensor *output, Tensor *target), I64 batch_size);
U0 BushTrain(Network *net, Tensor **inputs, Tensor **targets, Optimizer *opt, F64 (*loss_fn)(Tensor *output, Tensor *target), I64 epochs, I64 batch_size, Bool verbose);

// ===================================================
// Utilities
// ===================================================

Tensor *BushPredict(Network *net, Tensor *input);
F64 BushAccuracy(Network *net, Tensor **inputs, Tensor **targets);

I64 BushNetworkNumParams(Network *net);
Tensor **BushNetworkParams(Network *net, I64 *num_params);
U0 BushNetworkZeroGrad(Network *net);
U0 BushPrintNetwork(Network *net);

// ===================================================
// Auto
// ===================================================

Network *BushNetworkAuto(Tensor *input, Tensor *target); 