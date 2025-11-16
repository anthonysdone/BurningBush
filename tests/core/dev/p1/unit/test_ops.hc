U0 TestElementwiseOps() {
    "\n=== Test: Elementwise Operations ===\n";
    
    I64 shape[2] = {2, 2};
    F64 data_a[4] = {1.0, 2.0, 3.0, 4.0};
    F64 data_b[4] = {5.0, 6.0, 7.0, 8.0};
    
    Tensor *a = BushTensor(data_a, shape, 2);
    Tensor *b = BushTensor(data_b, shape, 2);
    
    // Test BushAdd
    Tensor *add = BushAdd(a, b);
    Bool add_ok = (add->data[0] == 6.0 && add->data[3] == 12.0);
    if (add_ok) "BushAdd: PASS\n";
    else "BushAdd: FAIL\n";
    BushRelease(add);
    
    // Test BushSub
    Tensor *sub = BushSub(a, b);
    Bool sub_ok = (sub->data[0] == -4.0 && sub->data[3] == -4.0);
    if (sub_ok) "BushSub: PASS\n";
    else "BushSub: FAIL\n";
    BushRelease(sub);
    
    // Test BushMul
    Tensor *mul = BushMul(a, b);
    Bool mul_ok = (mul->data[0] == 5.0 && mul->data[3] == 32.0);
    if (mul_ok) "BushMul: PASS\n";
    else "BushMul: FAIL\n";
    BushRelease(mul);
    
    // Test BushDiv
    Tensor *div = BushDiv(b, a);
    Bool div_ok = (div->data[0] == 5.0 && div->data[3] == 2.0);
    if (div_ok) "BushDiv: PASS\n";
    else "BushDiv: FAIL\n";
    BushRelease(div);
    
    BushRelease(a);
    BushRelease(b);
    
    "Elementwise Operations Tests Complete\n";
}

U0 TestLinearAlgebra() {
    "\n=== Test: Linear Algebra ===\n";
    
    // Test BushMatMul
    I64 shape_a[2] = {2, 3};
    I64 shape_b[2] = {3, 2};
    F64 data_a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    F64 data_b[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    Tensor *a = BushTensor(data_a, shape_a, 2);
    Tensor *b = BushTensor(data_b, shape_b, 2);
    
    Tensor *matmul = BushMatMul(a, b);
    Bool matmul_shape_ok = (matmul->shape[0] == 2 && matmul->shape[1] == 2);
    if (matmul_shape_ok) "BushMatMul shape: PASS\n";
    else "BushMatMul shape: FAIL\n";
    
    // First element should be 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    Bool matmul_val_ok = (matmul->data[0] == 22.0);
    if (matmul_val_ok) "BushMatMul values: PASS\n";
    else "BushMatMul values: FAIL\n";
    BushRelease(matmul);
    
    // Test BushT (transpose)
    Tensor *t = BushT(a);
    Bool transpose_ok = (t->shape[0] == 3 && t->shape[1] == 2 && t->data[0] == 1.0 && t->data[1] == 4.0);
    if (transpose_ok) "BushT (transpose): PASS\n";
    else "BushT (transpose): FAIL\n";
    BushRelease(t);
    
    BushRelease(a);
    BushRelease(b);
    
    "Linear Algebra Tests Complete\n";
}

U0 TestActivationFunctions() {
    "\n=== Test: Activation Functions ===\n";
    
    I64 shape[1] = {4};
    F64 data[4] = {-2.0, -1.0, 1.0, 2.0};
    Tensor *x = BushTensor(data, shape, 1);
    
    // Test BushReluTensor
    Tensor *relu = BushReluTensor(x);
    Bool relu_ok = (relu->data[0] == 0.0 && relu->data[1] == 0.0 && 
                    relu->data[2] == 1.0 && relu->data[3] == 2.0);
    if (relu_ok) "BushReluTensor: PASS\n";
    else "BushReluTensor: FAIL\n";
    BushRelease(relu);
    
    // Test BushSigmoidTensor
    Tensor *sigmoid = BushSigmoidTensor(x);
    Bool sigmoid_ok = (sigmoid->data[2] > 0.7 && sigmoid->data[2] < 0.8);
    if (sigmoid_ok) "BushSigmoidTensor: PASS\n";
    else "BushSigmoidTensor: FAIL\n";
    BushRelease(sigmoid);
    
    // Test BushTanhTensor
    Tensor *tanh_t = BushTanhTensor(x);
    Bool tanh_ok = (tanh_t->data[0] < 0.0 && tanh_t->data[3] > 0.0);
    if (tanh_ok) "BushTanhTensor: PASS\n";
    else "BushTanhTensor: FAIL\n";
    BushRelease(tanh_t);
    
    BushRelease(x);
    
    "Activation Functions Tests Complete\n";
}

U0 TestReductionOps() {
    "\n=== Test: Reduction Operations ===\n";
    
    I64 shape[2] = {2, 3};
    F64 data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Tensor *x = BushTensor(data, shape, 2);
    
    // Test BushSumAll
    Tensor *sum = BushSumAll(x);
    Bool sum_ok = (sum->size == 1 && sum->data[0] == 21.0);
    if (sum_ok) "BushSumAll: PASS\n";
    else "BushSumAll: FAIL\n";
    BushRelease(sum);
    
    // Test BushmeanAll
    Tensor *mean = BushmeanAll(x);
    Bool mean_ok = (mean->size == 1 && mean->data[0] == 3.5);
    if (mean_ok) "BushmeanAll: PASS\n";
    else "BushmeanAll: FAIL\n";
    BushRelease(mean);
    
    BushRelease(x);
    
    "Reduction Operations Tests Complete\n";
}

U0 TestBroadcastOps() {
    "\n=== Test: Broadcast Operations ===\n";
    
    I64 shape_a[2] = {2, 3};
    I64 shape_b[1] = {3};
    F64 data_a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    F64 data_b[3] = {10.0, 20.0, 30.0};
    
    Tensor *a = BushTensor(data_a, shape_a, 2);
    Tensor *b = BushTensor(data_b, shape_b, 1);
    
    // Test BushBroadcastAddInPlace
    BushBroadcastAddInPlace(a, b);
    Bool broadcast_ok = (a->data[0] == 11.0 && a->data[2] == 33.0 && a->data[5] == 36.0);
    if (broadcast_ok) "BushBroadcastAddInPlace: PASS\n";
    else "BushBroadcastAddInPlace: FAIL\n";
    
    BushRelease(a);
    BushRelease(b);
    
    "Broadcast Operations Tests Complete\n";
}

U0 RunOpsTests() {
    "\n=== Ops.hc Unit Tests ===\n";
    
    TestElementwiseOps();
    TestLinearAlgebra();
    TestActivationFunctions();
    TestReductionOps();
    TestBroadcastOps();
    
    "\n=== Ops.hc Unit Tests Complete ===\n";
}
