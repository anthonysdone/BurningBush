U0 TestMemoryAllocation() {
    "\n=== Test: Memory Allocation ===\n";
    
    // Test BushMalloc
    U0 *ptr1 = BushMalloc(100);
    if (ptr1) "BushMalloc: PASS\n";
    else "BushMalloc: FAIL\n";
    if (ptr1) BushFree(ptr1);
    
    // Test BushCalloc
    I64 *ptr2 = BushCalloc(10, sizeof(I64));
    if (ptr2) "BushCalloc: PASS\n";
    else "BushCalloc: FAIL\n";
    if (ptr2) {
        Bool all_zero = TRUE;
        for (I64 i = 0; i < 10; i++) {
            if (ptr2[i] != 0) all_zero = FALSE;
        }
        if (all_zero) "BushCalloc zeros memory: PASS\n";
        else "BushCalloc zeros memory: FAIL\n";
        BushFree(ptr2);
    }
    
    "Memory Allocation Tests Complete\n";
}

U0 TestShapeOperations() {
    "\n=== Test: Shape Operations ===\n";
    
    // Test BushShapeSize
    I64 shape1[3] = {2, 3, 4};
    I64 size1 = BushShapeSize(shape1, 3);
    if (size1 == 24) "BushShapeSize(2,3,4) = %d: PASS\n", size1;
    else "BushShapeSize(2,3,4) = %d: FAIL\n", size1;
    
    // Test BushShapeCopy
    I64 *shape_copy = BushShapeCopy(shape1, 3);
    Bool copy_ok = TRUE;
    for (I64 i = 0; i < 3; i++) {
        if (shape_copy[i] != shape1[i]) copy_ok = FALSE;
    }
    if (copy_ok) "BushShapeCopy: PASS\n";
    else "BushShapeCopy: FAIL\n";
    BushFree(shape_copy);
    
    // Test BushShapeEquals
    I64 shape2[3] = {2, 3, 4};
    I64 shape3[3] = {2, 3, 5};
    Bool eq1 = BushShapeEquals(shape1, shape2, 3, FALSE);
    Bool eq2 = BushShapeEquals(shape1, shape3, 3, FALSE);
    if (eq1) "BushShapeEquals (same): PASS\n";
    else "BushShapeEquals (same): FAIL\n";
    if (!eq2) "BushShapeEquals (different): PASS\n";
    else "BushShapeEquals (different): FAIL\n";
    
    // Test BushComputeStrides
    I64 *strides = BushComputeStrides(shape1, 3);
    Bool strides_ok = (strides[0] == 12 && strides[1] == 4 && strides[2] == 1);
    if (strides_ok) "BushComputeStrides: PASS\n";
    else "BushComputeStrides: FAIL\n";
    BushFree(strides);
    
    // Test BushComputeOffset
    I64 indices[3] = {1, 2, 3};
    I64 offset = BushComputeOffset(indices, strides, 3);
    "BushComputeOffset: computed\n";
    
    "Shape Operations Tests Complete\n";
}

U0 TestRandomGeneration() {
    "\n=== Test: Random Generation ===\n";
    
    // Test BushRandUniform
    F64 r1 = BushRandUniform(0.0, 10.0);
    Bool uniform_ok = (r1 >= 0.0 && r1 <= 10.0);
    if (uniform_ok) "BushRandUniform in range: PASS\n";
    else "BushRandUniform in range: FAIL\n";
    
    // Test BushRandNormal (just check it doesn't crash)
    F64 r2 = BushRandNormal(0.0, 1.0);
    "BushRandNormal: PASS (%.4f)\n", r2;
    
    // Test BushRandPerm
    I64 *perm = BushRandPerm(5);
    Bool has_all = TRUE;
    for (I64 i = 0; i < 5; i++) {
        Bool found = FALSE;
        for (I64 j = 0; j < 5; j++) {
            if (perm[j] == i) found = TRUE;
        }
        if (!found) has_all = FALSE;
    }
    if (has_all) "BushRandPerm contains all elements: PASS\n";
    else "BushRandPerm contains all elements: FAIL\n";
    BushFree(perm);
    
    "Random Generation Tests Complete\n";
}

U0 TestMathUtilities() {
    "\n=== Test: Math Utilities ===\n";
    
    // Test BushSigmoid
    F64 sig0 = BushSigmoid(0.0);
    Bool sig_ok = (sig0 > 0.49 && sig0 < 0.51);
    if (sig_ok) "BushSigmoid(0) ≈ 0.5: PASS\n";
    else "BushSigmoid(0) ≈ 0.5: FAIL\n";
    
    // Test BushRelu
    F64 relu_pos = BushRelu(5.0);
    F64 relu_neg = BushRelu(-5.0);
    Bool relu_ok = (relu_pos == 5.0 && relu_neg == 0.0);
    if (relu_ok) "BushRelu: PASS\n";
    else "BushRelu: FAIL\n";
    
    // Test BushTanh
    F64 tanh0 = BushTanh(0.0);
    Bool tanh_ok = (tanh0 > -0.01 && tanh0 < 0.01);
    if (tanh_ok) "BushTanh(0) ≈ 0: PASS\n";
    else "BushTanh(0) ≈ 0: FAIL\n";
    
    "Math Utilities Tests Complete\n";
}

U0 RunUtilTests() {
    "\n=== Util.hc Unit Tests ===\n";
    
    TestMemoryAllocation();
    TestShapeOperations();
    TestRandomGeneration();
    TestMathUtilities();
    
    "\n=== Util.hc Unit Tests Complete ===\n";
}
