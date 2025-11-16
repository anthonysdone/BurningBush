#include "core/Util.hc"

U0 Test() {
    "\n=== Test: Basic Memory Allocation ===\n";
    
    // Test 1: BushMalloc
    U0 *ptr1 = BushMalloc(100);
    if (ptr1) {
        "✓ BushMalloc(100): PASS\n";
        BushFree(ptr1);
    } else {
        "✗ BushMalloc(100): FAIL\n";
    }
    
    // Test 2: BushCalloc
    I64 *ptr2 = BushCalloc(10, sizeof(I64));
    if (ptr2) {
        Bool all_zero = TRUE;
        for (I64 i = 0; i < 10; i++) {
            if (ptr2[i] != 0) all_zero = FALSE;
        }
        if (all_zero) {
            "✓ BushCalloc zeros memory: PASS\n";
        } else {
            "✗ BushCalloc zeros memory: FAIL\n";
        }
        BushFree(ptr2);
    } else {
        "✗ BushCalloc(10, 8): FAIL\n";
    }
    
    "\n=== Test Complete ===\n";
}

Test();
