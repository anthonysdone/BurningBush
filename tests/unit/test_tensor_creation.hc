#include "core/Util.hc"
#include "core/Tensor.hc"

U0 Test() {
    "\n=== Test: Tensor Creation ===\n";
    
    // Test 1: BushZeros
    I64 shape1[2];
    shape1[0] = 2;
    shape1[1] = 3;
    
    Tensor *t1 = BushZeros(shape1, 2);
    if (t1) {
        Bool all_zero = TRUE;
        for (I64 i = 0; i < t1->size; i++) {
            if (t1->data[i] != 0.0) {
                all_zero = FALSE;
            }
        }
        if (all_zero && t1->size == 6) {
            "✓ BushZeros(2,3): PASS\n";
        } else {
            "✗ BushZeros(2,3): FAIL\n";
        }
        BushRelease(t1);
    } else {
        "✗ BushZeros(2,3): FAIL (null)\n";
    }
    
    // Test 2: BushOnes
    I64 shape2[2];
    shape2[0] = 3;
    shape2[1] = 2;
    
    Tensor *t2 = BushOnes(shape2, 2);
    if (t2) {
        Bool all_one = TRUE;
        for (I64 i = 0; i < t2->size; i++) {
            if (t2->data[i] != 1.0) {
                all_one = FALSE;
            }
        }
        if (all_one && t2->size == 6) {
            "✓ BushOnes(3,2): PASS\n";
        } else {
            "✗ BushOnes(3,2): FAIL\n";
        }
        BushRelease(t2);
    } else {
        "✗ BushOnes(3,2): FAIL (null)\n";
    }
    
    // Test 3: BushEmpty
    I64 shape3[1];
    shape3[0] = 5;
    
    Tensor *t3 = BushEmpty(shape3, 1);
    if (t3 && t3->size == 5) {
        "✓ BushEmpty(5): PASS\n";
        BushRelease(t3);
    } else {
        "✗ BushEmpty(5): FAIL\n";
    }
    
    "\n=== Test Complete ===\n";
}

Test();
