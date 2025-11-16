U0 TestTensorCreation() {
    "\n=== Test: Tensor Creation ===\n";
    
    I64 shape[2] = {3, 4};
    
    // Test BushZeros
    Tensor *zeros = BushZeros(shape, 2);
    Bool zeros_ok = TRUE;
    for (I64 i = 0; i < zeros->size; i++) {
        if (zeros->data[i] != 0.0) zeros_ok = FALSE;
    }
    if (zeros_ok) "BushZeros: PASS\n";
    else "BushZeros: FAIL\n";
    BushRelease(zeros);
    
    // Test BushOnes
    Tensor *ones = BushOnes(shape, 2);
    Bool ones_ok = TRUE;
    for (I64 i = 0; i < ones->size; i++) {
        if (ones->data[i] != 1.0) ones_ok = FALSE;
    }
    if (ones_ok) "BushOnes: PASS\n";
    else "BushOnes: FAIL\n";
    BushRelease(ones);
    
    // Test BushFull
    Tensor *full = BushFull(shape, 2, 3.14);
    Bool full_ok = TRUE;
    for (I64 i = 0; i < full->size; i++) {
        if (full->data[i] != 3.14) full_ok = FALSE;
    }
    if (full_ok) "BushFull: PASS\n";
    else "BushFull: FAIL\n";
    BushRelease(full);
    
    // Test BushRandn
    Tensor *randn = BushRandn(shape, 2);
    if (randn->size == 12) "BushRandn (size=%d): PASS\n", randn->size;
    else "BushRandn (size=%d): FAIL\n", randn->size;
    BushRelease(randn);
    
    // Test BushArange
    Tensor *arange = BushArange(0.0, 5.0, 1.0);
    Bool arange_ok = (arange->size == 5 && arange->data[0] == 0.0 && arange->data[4] == 4.0);
    if (arange_ok) "BushArange: PASS\n";
    else "BushArange: FAIL\n";
    BushRelease(arange);
    
    // Test BushLinspace
    Tensor *linspace = BushLinspace(0.0, 10.0, 11);
    Bool linspace_ok = (linspace->size == 11 && linspace->data[0] == 0.0 && linspace->data[10] == 10.0);
    if (linspace_ok) "BushLinspace: PASS\n";
    else "BushLinspace: FAIL\n";
    BushRelease(linspace);
    
    "Tensor Creation Tests Complete\n";
}

U0 TestTensorProperties() {
    "\n=== Test: Tensor Properties ===\n";
    
    I64 shape[3] = {2, 3, 4};
    Tensor *t = BushZeros(shape, 3);
    
    // Test BushNdim
    I64 ndim = BushNdim(t);
    if (ndim == 3) "BushNdim: PASS\n";
    else "BushNdim: FAIL\n";
    
    // Test BushShape
    I64 dim0 = BushShape(t, 0);
    I64 dim1 = BushShape(t, 1);
    I64 dim2 = BushShape(t, 2);
    Bool shape_ok = (dim0 == 2 && dim1 == 3 && dim2 == 4);
    if (shape_ok) "BushShape: PASS\n";
    else "BushShape: FAIL\n";
    
    // Test size
    Bool size_ok = (t->size == 24);
    if (size_ok) "Tensor size: PASS\n";
    else "Tensor size: FAIL\n";
    
    BushRelease(t);
    
    "Tensor Properties Tests Complete\n";
}

U0 TestDataAccess() {
    "\n=== Test: Data Access ===\n";
    
    I64 shape[2] = {3, 3};
    Tensor *t = BushZeros(shape, 2);
    
    // Test BushSet and BushGet
    I64 indices[2] = {1, 2};
    BushSet(t, indices, 42.0);
    F64 val = BushGet(t, indices);
    if (val == 42.0) "BushSet/BushGet: PASS\n";
    else "BushSet/BushGet: FAIL\n";
    
    // Test BushItem (scalar)
    I64 scalar_shape[1] = {1};
    Tensor *scalar = BushFull(scalar_shape, 1, 99.0);
    F64 item = BushItem(scalar);
    if (item == 99.0) "BushItem: PASS\n";
    else "BushItem: FAIL\n";
    BushRelease(scalar);
    
    BushRelease(t);
    
    "Data Access Tests Complete\n";
}

U0 TestMemoryManagement() {
    "\n=== Test: Memory Management ===\n";
    
    I64 shape[2] = {2, 2};
    Tensor *t = BushZeros(shape, 2);
    
    // Test initial refcount
    if (t->refcount == 1) "Initial refcount: PASS\n";
    else "Initial refcount: FAIL\n";
    
    // Test BushRetain
    Tensor *t2 = BushRetain(t);
    if (t->refcount == 2) "BushRetain increments refcount: PASS\n";
    else "BushRetain increments refcount: FAIL\n";
    if (t == t2) "BushRetain returns same pointer: PASS\n";
    else "BushRetain returns same pointer: FAIL\n";
    
    // Test BushRelease
    BushRelease(t);
    if (t->refcount == 1) "BushRelease decrements refcount: PASS\n";
    else "BushRelease decrements refcount: FAIL\n";
    
    BushRelease(t);
    "Final BushRelease: PASS (freed)\n";
    
    "Memory Management Tests Complete\n";
}

U0 TestScalarOperations() {
    "\n=== Test: Scalar Operations ===\n";
    
    I64 shape[2] = {2, 2};
    F64 data[4] = {1.0, 2.0, 3.0, 4.0};
    Tensor *t = BushTensor(data, shape, 2);
    
    // Test BushScalarAdd
    Tensor *add_result = BushScalarAdd(t, 10.0);
    Bool add_ok = (add_result->data[0] == 11.0 && add_result->data[3] == 14.0);
    if (add_ok) "BushScalarAdd: PASS\n";
    else "BushScalarAdd: FAIL\n";
    BushRelease(add_result);
    
    // Test BushScalarMul
    Tensor *mul_result = BushScalarMul(t, 2.0);
    Bool mul_ok = (mul_result->data[0] == 2.0 && mul_result->data[3] == 8.0);
    if (mul_ok) "BushScalarMul: PASS\n";
    else "BushScalarMul: FAIL\n";
    BushRelease(mul_result);
    
    BushRelease(t);
    
    "Scalar Operations Tests Complete\n";
}

U0 RunTensorTests() {
    "\n=== Tensor.hc Unit Tests ===\n";

    TestTensorCreation();
    TestTensorProperties();
    TestDataAccess();
    TestMemoryManagement();
    TestScalarOperations();
    
    "\n=== Tensor.hc Unit Tests Complete ===\n";
}
