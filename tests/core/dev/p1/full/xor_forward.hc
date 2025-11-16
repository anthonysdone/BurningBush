U0 TestXORForward() {
    "\n=== XOR Forward Test ===\n";

    "\n--- Loading Data ---\n";

    F64 x_data [4][2] = {
        {0.0, 0.0}, 
        {0.0, 1.0}, 
        {1.0, 0.0}, 
        {1.0, 1.0}
    }; 

    F64 y_data [4][1] = {
        {0.0}, 
        {1.0}, 
        {1.0}, 
        {0.0}
    };

    I64 x_shape[2] = {4, 2};
    Tensor *X = BushTensor(&x_data[0][0], x_shape, 2);

    "Input (X): \n"; 
    BushPrint(X);

    I64 y_shape[2] = {4, 1};
    Tensor *Y = BushTensor(&y_data[0][0], y_shape, 2);

    "Target (Y): \n";
    BushPrint(Y);

    "\n--- Building Model ---\n";

    Module *modules[4]; 
    modules[0] = BushLinear(2, 4, TRUE);
    modules[1] = BushTanhModule(); 
    modules[2] = BushLinear(4, 1, TRUE); 
    modules[3] = BushSigmoidModule();

    Module *model = BushSequential(modules, 4);

    "Parameters: %d\n", BushNumParameters(model); 

    "\n--- Forward Pass ---\n";
    Tensor *pred = BushForward(model, X); 

    "Prediction: \n";
    BushPrint(pred); 

    Tensor *diff = BushSub(pred, Y);
    Tensor *squared = BushMul(diff, diff);
    Tensor *loss = BushmeanAll(squared);

    "Loss: \n";
    BushPrint(loss);

    BushRelease(X);
    BushRelease(Y);
    BushRelease(pred);
    BushRelease(diff);
    BushRelease(squared);
    BushRelease(loss);

    "\n=== XOR Forward Test Complete ===\n";
}

TestXORForward();