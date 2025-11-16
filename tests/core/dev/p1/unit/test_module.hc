U0 TestModuleCreation() {
    "\n-- Test: Module Creation --\n";
    
    // Test BushLinear
    Module *linear = BushLinear(3, 5, TRUE);
    Bool linear_ok = (linear->module_type == MODULE_LINEAR && 
                      linear->num_params == 2 &&
                      linear->params[0]->shape[0] == 5 &&
                      linear->params[0]->shape[1] == 3);
    if (linear_ok) "BushLinear: PASS\n";
    else "BushLinear: FAIL\n";
    
    // Test BushLinear without bias
    Module *linear_no_bias = BushLinear(3, 5, FALSE);
    Bool no_bias_ok = (linear_no_bias->num_params == 1);
    if (no_bias_ok) "BushLinear (no bias): PASS\n";
    else "BushLinear (no bias): FAIL\n";
    
    // Test BushReluModule
    Module *relu = BushReluModule();
    Bool relu_ok = (relu->module_type == MODULE_RELU && relu->num_params == 0);
    if (relu_ok) "BushReluModule: PASS\n";
    else "BushReluModule: FAIL\n";
    
    // Test BushSigmoidModule
    Module *sigmoid = BushSigmoidModule();
    Bool sigmoid_ok = (sigmoid->module_type == MODULE_SIGMOID);
    if (sigmoid_ok) "BushSigmoidModule: PASS\n";
    else "BushSigmoidModule: FAIL\n";
    
    // Test BushTanhModule
    Module *tanh_m = BushTanhModule();
    Bool tanh_ok = (tanh_m->module_type == MODULE_TANH);
    if (tanh_ok) "BushTanhModule: PASS\n";
    else "BushTanhModule: FAIL\n";
    
    "Module Creation Tests Complete\n";
}

U0 TestSequentialModule() {
    "\n--- Test: Sequential Module ---\n";
    
    Module *modules[3];
    modules[0] = BushLinear(2, 4, TRUE);
    modules[1] = BushReluModule();
    modules[2] = BushLinear(4, 1, FALSE);
    
    Module *seq = BushSequential(modules, 3);
    Bool seq_ok = (seq->module_type == MODULE_SEQUENTIAL);
    if (seq_ok) "BushSequential creation: PASS\n";
    else "BushSequential creation: FAIL\n";
    
    Sequential *seq_data = seq->extra;
    Bool seq_data_ok = (seq_data->num_modules == 3);
    if (seq_data_ok) "Sequential num_modules: PASS\n";
    else "Sequential num_modules: FAIL\n";
    
    "Sequential Module Tests Complete\n";
}

U0 TestLinearForward() {
    "\n--- Test: Linear Forward Pass ---\n";
    
    Module *linear = BushLinear(2, 3, FALSE);
    
    // Set weights to identity-like for predictable output
    for (I64 i = 0; i < linear->params[0]->size; i++) {
        linear->params[0]->data[i] = 1.0;
    }
    
    // Create input
    I64 input_shape[2] = {1, 2};
    F64 input_data[2] = {1.0, 2.0};
    Tensor *input = BushTensor(input_data, input_shape, 2);
    
    // Forward pass
    Tensor *output = BushForward(linear, input);
    
    Bool forward_ok = (output->shape[0] == 1 && output->shape[1] == 3);
    if (forward_ok) "Linear forward shape: PASS\n";
    else "Linear forward shape: FAIL\n";
    
    BushRelease(input);
    BushRelease(output);
    
    "Linear Forward Tests Complete\n";
}

U0 TestActivationForward() {
    "\n--- Test: Activation Forward Pass ---\n";
    
    I64 input_shape[2] = {2, 2};
    F64 input_data[4] = {-1.0, -0.5, 0.5, 1.0};
    Tensor *input = BushTensor(input_data, input_shape, 2);
    
    // Test ReLU forward
    Module *relu = BushReluModule();
    Tensor *relu_out = BushForward(relu, input);
    Bool relu_ok = (relu_out->data[0] == 0.0 && relu_out->data[3] == 1.0);
    if (relu_ok) "ReLU forward: PASS\n";
    else "ReLU forward: FAIL\n";
    BushRelease(relu_out);
    
    // Test Sigmoid forward
    Module *sigmoid = BushSigmoidModule();
    Tensor *sigmoid_out = BushForward(sigmoid, input);
    Bool sigmoid_ok = (sigmoid_out->data[0] < 0.5 && sigmoid_out->data[3] > 0.5);
    if (sigmoid_ok) "Sigmoid forward: PASS\n";
    else "Sigmoid forward: FAIL\n";
    BushRelease(sigmoid_out);
    
    // Test Tanh forward
    Module *tanh_m = BushTanhModule();
    Tensor *tanh_out = BushForward(tanh_m, input);
    Bool tanh_ok = (tanh_out->data[0] < 0.0 && tanh_out->data[3] > 0.0);
    if (tanh_ok) "Tanh forward: PASS\n";
    else "Tanh forward: FAIL\n";
    BushRelease(tanh_out);
    
    BushRelease(input);
    
    "Activation Forward Tests Complete\n";
}

U0 TestSequentialForward() {
    "\n--- Test: Sequential Forward Pass ---\n";
    
    Module *modules[2];
    modules[0] = BushLinear(2, 3, FALSE);
    modules[1] = BushReluModule();
    
    // Set weights
    for (I64 i = 0; i < modules[0]->params[0]->size; i++) {
        modules[0]->params[0]->data[i] = 0.5;
    }
    
    Module *seq = BushSequential(modules, 2);
    
    // Create input
    I64 input_shape[2] = {1, 2};
    F64 input_data[2] = {1.0, 1.0};
    Tensor *input = BushTensor(input_data, input_shape, 2);
    
    // Forward pass
    Tensor *output = BushForward(seq, input);
    
    Bool seq_forward_ok = (output->shape[0] == 1 && output->shape[1] == 3);
    if (seq_forward_ok) "Sequential forward shape: PASS\n";
    else "Sequential forward shape: FAIL\n";
    
    BushRelease(input);
    BushRelease(output);
    
    "Sequential Forward Tests Complete\n";
}

U0 TestParameterManagement() {
    "\n--- Test: Parameter Management ---\n";
    
    Module *modules[3];
    modules[0] = BushLinear(2, 4, TRUE);  // 2*4 weights + 4 bias = 12 params
    modules[1] = BushReluModule();        // 0 params
    modules[2] = BushLinear(4, 1, TRUE);  // 4*1 weights + 1 bias = 5 params
    
    Module *seq = BushSequential(modules, 3);
    
    // Test BushNumParameters
    I64 num_params = BushNumParameters(seq);
    Bool num_params_ok = (num_params >= 2);  // 2 parameter tensors from first linear
    if (num_params >= 2) "BushNumParameters count: PASS (%d params)\n", num_params;
    else "BushNumParameters count: FAIL (%d params)\n", num_params;
    
    // Test BushCountParameters
    I64 count = BushCountParameters(seq);
    "BushCountParameters: %d total\n", count;
    
    "Parameter Management Tests Complete\n";
}

U0 TestModuleTypeDispatch() {
    "\n--- Test: Module Type Dispatch ---\n";
    
    I64 input_shape[2] = {1, 2};
    F64 input_data[2] = {1.0, 2.0};
    Tensor *input = BushTensor(input_data, input_shape, 2);
    
    // Test each module type dispatch
    Module *linear = BushLinear(2, 2, FALSE);
    Tensor *out1 = BushForward(linear, input);
    if (out1) "Linear dispatch: PASS\n";
    else "Linear dispatch: FAIL\n";
    BushRelease(out1);
    
    Module *relu = BushReluModule();
    Tensor *out2 = BushForward(relu, input);
    if (out2) "ReLU dispatch: PASS\n";
    else "ReLU dispatch: FAIL\n";
    BushRelease(out2);
    
    Module *sigmoid = BushSigmoidModule();
    Tensor *out3 = BushForward(sigmoid, input);
    if (out3) "Sigmoid dispatch: PASS\n";
    else "Sigmoid dispatch: FAIL\n";
    BushRelease(out3);
    
    Module *tanh_m = BushTanhModule();
    Tensor *out4 = BushForward(tanh_m, input);
    if (out4) "Tanh dispatch: PASS\n";
    else "Tanh dispatch: FAIL\n";
    BushRelease(out4);
    
    BushRelease(input);
    
    "Module Type Dispatch Tests Complete\n";
}

U0 RunModuleTests() {
    "\n=== Module.hc Unit Tests ===\n";
    
    TestModuleCreation();
    TestSequentialModule();
    TestLinearForward();
    TestActivationForward();
    TestSequentialForward();
    TestParameterManagement();
    TestModuleTypeDispatch();

    "\n=== Module.hc Unit Tests Complete ===\n";
}
