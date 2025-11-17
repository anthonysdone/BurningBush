// ===================================================
// Memory Utilities
// ===================================================

U0 *BushAlloc(I64 size); 
U0 BushFree(U0 *ptr);
U0 *BushRealloc(U0 *ptr, I64 new_size);

// ===================================================
// Math Utilities
// ===================================================

F64 BushRandF64(I64 seed, F64 min, F64 max); 
F64 BushRandI64(I64 seed, I64 min, I64 max);

F64 BushExp(F64 x); 
F64 BushLog(F64 x);
F64 BushSqrt(F64 x);
F64 BushCos(F64 x);

// ===================================================
// Shape Utilities
// ===================================================

I64 BushShapeSize(I64 *shape, I64 ndim);
Bool BushShapeEqual(I64 *shape1, I64 *shape2, I64 ndim);

// ===================================================
// Debug Utilities
// ===================================================

U0 BushPrint(U8 *message);
U0 BushError(U8 *message);
U0 BushWarning(U8 *message);
U0 BushAssert(Bool condition, U8 *message);