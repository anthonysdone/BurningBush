#include "core/Util.hc"

U0 Test() {
    "\n=== Test: Shape Operations ===\n";
    
    // Test 1: BushShapeSize
    I64 shape1[3];
    shape1[0] = 2;
    shape1[1] = 3;
    shape1[2] = 4;
    
    I64 size1 = BushShapeSize(shape1, 3);
    if (size1 == 24) {
        "✓ BushShapeSize(2,3,4) = 24: PASS\n";
    } else {
        "✗ BushShapeSize(2,3,4) = %lld: FAIL (expected 24)\n", size1;
    }
    
    // Test 2: BushShapeCopy
    I64 *shape_copy = BushShapeCopy(shape1, 3);
    Bool copy_ok = TRUE;
    for (I64 i = 0; i < 3; i++) {
        if (shape_copy[i] != shape1[i]) {
            copy_ok = FALSE;
        }
    }
    if (copy_ok) {
        "✓ BushShapeCopy: PASS\n";
    } else {
        "✗ BushShapeCopy: FAIL\n";
    }
    BushFree(shape_copy);
    
    // Test 3: BushShapeEquals
    I64 shape2[3];
    shape2[0] = 2;
    shape2[1] = 3;
    shape2[2] = 4;
    
    I64 shape3[3];
    shape3[0] = 2;
    shape3[1] = 3;
    shape3[2] = 5;
    
    Bool eq1 = BushShapeEquals(shape1, shape2, 3, FALSE);
    Bool eq2 = BushShapeEquals(shape1, shape3, 3, FALSE);
    
    if (eq1 && !eq2) {
        "✓ BushShapeEquals: PASS\n";
    } else {
        "✗ BushShapeEquals: FAIL\n";
    }
    
    "\n=== Test Complete ===\n";
}

Test();
