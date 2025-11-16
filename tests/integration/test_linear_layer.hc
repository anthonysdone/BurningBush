#include "../../core/Util.hc"
#include "../../core/Tensor.hc"
#include "../../core/Ops.hc"
#include "../../core/Module.hc"

U0 Test() {
    "\n=== Test: Linear Layer (KNOWN BUG - WILL TIMEOUT) ===\n";
    
    "Attempting to create Linear(2, 4, FALSE)...\n";
    Module *lin = BushLinear(2, 4, FALSE);
    
    if (lin) {
        "✓ Linear module created\n";
        "  Parameters: %lld\n", lin->num_params;
    } else {
        "✗ Linear module creation failed\n";
    }
    
    "\n=== Test Complete ===\n";
}

Test();
