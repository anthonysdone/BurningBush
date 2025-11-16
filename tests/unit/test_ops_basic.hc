#include "../../core/Util.hc"
#include "../../core/Tensor.hc"
#include "../../core/Ops.hc"

U0 Test() {
    "\n=== Test: Basic Operations ===\n";
    
    // Create test tensors
    I64 shape[2];
    shape[0] = 2;
    shape[1] = 2;
    
    Tensor *a = BushOnes(shape, 2);
    Tensor *b = BushOnes(shape, 2);
    
    // Test 1: Addition
    Tensor *sum = BushAdd(a, b);
    if (sum) {
        Bool correct = TRUE;
        for (I64 i = 0; i < sum->size; i++) {
            if (sum->data[i] != 2.0) {
                correct = FALSE;
            }
        }
        if (correct) {
            "✓ BushAdd: PASS\n";
        } else {
            "✗ BushAdd: FAIL\n";
        }
        BushRelease(sum);
    } else {
        "✗ BushAdd: FAIL (null)\n";
    }
    
    // Test 2: Multiplication
    Tensor *prod = BushMul(a, b);
    if (prod) {
        Bool correct = TRUE;
        for (I64 i = 0; i < prod->size; i++) {
            if (prod->data[i] != 1.0) {
                correct = FALSE;
            }
        }
        if (correct) {
            "✓ BushMul: PASS\n";
        } else {
            "✗ BushMul: FAIL\n";
        }
        BushRelease(prod);
    } else {
        "✗ BushMul: FAIL (null)\n";
    }
    
    BushRelease(a);
    BushRelease(b);
    
    "\n=== Test Complete ===\n";
}

Test();
