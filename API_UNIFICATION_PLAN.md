# API Unification Plan: Bringing Runtime and Compile-Time Expression APIs Closer

## Executive Summary

This plan addresses the API differences between the runtime (`ExpressionBuilder`/`TypedBuilderExpr`) and compile-time (`ScopedMathExpr`/`ScopeBuilder`) expression building systems. The most critical issue discovered is that **the compile-time system is hardcoded to f64**, violating the core requirement of "generic but strongly typed" design.

## Critical Issue: Type System Limitations

### **BLOCKER: Compile-Time System Hardcoded to f64**

The compile-time system currently has a fundamental limitation:

```rust
// Current compile-time API - HARDCODED TO f64
pub trait ScopedMathExpr<const SCOPE: usize>: Clone + Sized {
    fn eval(&self, vars: &ScopedVarArray<SCOPE>) -> f64;  // ❌ f64 only
    fn to_ast(&self) -> ASTRepr<f64>;                     // ❌ f64 only
}

struct ScopedVarArray<const SCOPE: usize> {
    vars: Vec<f64>,  // ❌ f64 only
}
```

**vs Runtime system - properly generic:**

```rust
// Runtime API - properly generic
pub struct TypedBuilderExpr<T> {
    ast: ASTRepr<T>,  // ✅ Generic over T: NumericType
}

impl<T: NumericType> TypedBuilderExpr<T> { ... }
```

## Corrected Priority Order

### **Phase 0: Fix Type System Architecture (CRITICAL)**

Make the compile-time system generic over numeric types to match the runtime system:

```rust
// Fixed compile-time API - generic but strongly typed
pub trait ScopedMathExpr<T, const SCOPE: usize>: Clone + Sized 
where
    T: NumericType,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T;
    fn to_ast(&self) -> ASTRepr<T>;
    
    // Mathematical operations remain strongly typed
    fn add<U>(self, other: U) -> ScopedAdd<T, Self, U, SCOPE> 
    where 
        U: ScopedMathExpr<T, SCOPE>;
}

// Generic variable array
pub struct ScopedVarArray<T, const SCOPE: usize> 
where
    T: NumericType,
{
    vars: Vec<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

// Generic variables and constants  
pub struct ScopedVar<T, const ID: usize, const SCOPE: usize>(PhantomData<T>);
pub struct ScopedConst<T, const BITS: u64, const SCOPE: usize>(PhantomData<T>);
```

### **Phase 1: Add Operator Overloading to Compile-Time API**

Once generic, add operator overloading:

```rust
impl<T, L, R, const SCOPE: usize> Add<R> for L
where
    T: NumericType + Add<Output = T>,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    type Output = ScopedAdd<T, L, R, SCOPE>;
    
    fn add(self, rhs: R) -> Self::Output {
        ScopedMathExpr::add(self, rhs)
    }
}
```

### **Phase 2: Harmonize Method Names and Builder Names**

```rust
// Unified naming
pub struct MathBuilder { ... }           // Runtime builder
pub struct StaticMathBuilder { ... }     // Compile-time builder

// Both support:
builder.var()        // Create variable
builder.constant()   // Create constant
```

## Technical Implementation Plan

### Step 1: Make ScopedMathExpr Generic

```rust
// New generic trait
pub trait ScopedMathExpr<T, const SCOPE: usize>: Clone + Sized 
where
    T: NumericType,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T;
    fn to_ast(&self) -> ASTRepr<T>;
    
    // Keep existing method-based API for now
    fn add<U>(self, other: U) -> ScopedAdd<T, Self, U, SCOPE> 
    where U: ScopedMathExpr<T, SCOPE>;
    
    fn mul<U>(self, other: U) -> ScopedMul<T, Self, U, SCOPE> 
    where U: ScopedMathExpr<T, SCOPE>;
    
    // ... other operations
}
```

### Step 2: Update All Expression Types

```rust
// Generic operation types
pub struct ScopedAdd<T, L, R, const SCOPE: usize> 
where
    T: NumericType,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    left: L,
    right: R,
    _type: PhantomData<T>,
    _scope: PhantomData<[(); SCOPE]>,
}

impl<T, L, R, const SCOPE: usize> ScopedMathExpr<T, SCOPE> for ScopedAdd<T, L, R, SCOPE>
where
    T: NumericType + Add<Output = T>,
    L: ScopedMathExpr<T, SCOPE>,
    R: ScopedMathExpr<T, SCOPE>,
{
    fn eval(&self, vars: &ScopedVarArray<T, SCOPE>) -> T {
        self.left.eval(vars) + self.right.eval(vars)
    }
    
    fn to_ast(&self) -> ASTRepr<T> {
        ASTRepr::Add(Box::new(self.left.to_ast()), Box::new(self.right.to_ast()))
    }
}
```

### Step 3: Update Builder APIs

```rust
pub struct ScopeBuilder<T, const SCOPE: usize, const NEXT_ID: usize> 
where
    T: NumericType,
{
    _type: PhantomData<T>,
}

impl<T, const SCOPE: usize, const NEXT_ID: usize> ScopeBuilder<T, SCOPE, NEXT_ID>
where
    T: NumericType,
{
    pub fn var(self) -> (ScopedVar<T, NEXT_ID, SCOPE>, ScopeBuilder<T, SCOPE, { NEXT_ID + 1 }>) {
        (ScopedVar(PhantomData), ScopeBuilder { _type: PhantomData })
    }
    
    pub fn constant(self, value: T) -> ScopedConstValue<T, SCOPE> {
        ScopedConstValue { value, _type: PhantomData, _scope: PhantomData }
    }
}
```

## Migration Strategy

### Backward Compatibility
- Keep the old f64-only API as deprecated aliases
- Add new generic API alongside
- Provide migration guide

### Testing Strategy  
- Port all existing tests to use generic API
- Add tests for f32 and other numeric types
- Ensure compilation performance isn't degraded

## Benefits After Implementation

1. **True Type Safety**: Both APIs will be generic but strongly typed
2. **API Consistency**: Similar patterns between runtime and compile-time
3. **Feature Parity**: Both systems support the same numeric types
4. **Better Ergonomics**: Operator overloading in both systems

## Next Steps

1. **URGENT: Fix the type system** - Make compile-time system generic
2. **Add operator overloading** - Bring ergonomics up to runtime level  
3. **Harmonize naming** - Consistent builder and method names
4. **Update documentation** - Show equivalent patterns

This architectural fix is the foundation for true API unification. 