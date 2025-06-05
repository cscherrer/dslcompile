# Unified API Design: Two-Interface Strategy

## Vision: Only Two Interfaces, Both Heterogeneous by Default

Based on user feedback, the goal is to have **ONLY** two interfaces that users need to understand:

1. **`StaticContext`** - Compile-time optimization, zero overhead
2. **`DynamicContext`** - Runtime flexibility, minimal overhead

Both interfaces assume **heterogeneity by default** with no additional overhead when possible.

## Current State Analysis

### Current API Divergence

**Static Context (compile-time)**:
```rust
let mut ctx = Context::new_f64();
let expr = ctx.new_scope(|scope| {
    let (x, scope) = scope.auto_var();
    let (y, _scope) = scope.auto_var();
    x + y
});
```

**Dynamic Context (runtime)**:
```rust
let ctx = DynamicContext::new();
let x = ctx.var();
let y = ctx.var();
let expr = &x + &y;
```

**Heterogeneous Context (compile-time)**:
```rust
let mut ctx = HeteroContext::<0, 8>::new();
let x = ctx.var::<f64>();
let y = ctx.var::<Vec<f64>>();
let expr = hetero_add(x, y);
```

### Problems with Current Approach

1. **User choice paralysis**: Users must understand when to use "hetero" vs regular
2. **API fragmentation**: Three different syntaxes for essentially the same operations
3. **Mental overhead**: Understanding scopes, const generics, and type parameters
4. **Inconsistent interfaces**: Different method names and patterns

## Unified Design

### Core Principle: Hide Implementation Details

Users should only need to know:
- **Static = compile-time optimized**
- **Dynamic = runtime flexible**

Both support the same operations transparently.

### Unified API Surface

```rust
// STATIC CONTEXT - Compile-time optimized
let ctx = StaticContext::new();
let x = ctx.var::<f64>();           // Scalar variable
let data = ctx.var::<Vec<f64>>();   // Array variable  
let index = ctx.var::<usize>();     // Index variable
let result = ctx.eval(&expr, inputs);

// DYNAMIC CONTEXT - Runtime flexible
let ctx = DynamicContext::new();
let x = ctx.var::<f64>();           // Scalar variable
let data = ctx.var::<Vec<f64>>();   // Array variable
let index = ctx.var::<usize>();     // Index variable  
let result = ctx.eval(&expr, inputs);
```

### Key Design Elements

#### 1. Identical Method Names
```rust
// Both contexts support the same operations
ctx.var::<T>()           // Create typed variable
ctx.constant(value)      // Create constant
ctx.eval(expr, inputs)   // Evaluate expression
ctx.sum(range, |i| ...)  // Summation operations
```

#### 2. Transparent Type Handling
```rust
// Users don't choose "hetero" - it's automatic
let x = ctx.var::<f64>();        // Works in both contexts
let array = ctx.var::<Vec<f64>>();  // Works in both contexts
let expr = x + array[index];     // Works in both contexts
```

#### 3. Unified Input/Output
```rust
// Same input format for both contexts
let inputs = Inputs::new()
    .add("x", 3.0)
    .add("data", vec![1.0, 2.0, 3.0])
    .add("index", 1);

let result = ctx.eval(&expr, inputs);  // Same for both
```

#### 4. Zero-Overhead Heterogeneity
```rust
// No runtime dispatch when types are known at compile time
let expr = x + constant(2.0);  // Monomorphized to f64 + f64
let elem = array[index];       // Monomorphized to Vec<f64>[usize]
```

## Implementation Strategy

### Phase 1: DynamicContext Heterogeneous Enhancement

**Goal**: Make `DynamicContext` support heterogeneous types transparently

```rust
impl DynamicContext {
    /// Unified variable creation (type-aware)
    pub fn var<T: ExprType>(&self) -> Var<T> {
        // Automatically handles f64, Vec<f64>, usize, etc.
    }
    
    /// Unified evaluation with heterogeneous inputs
    pub fn eval<R>(&self, expr: &Expr<R>, inputs: &Inputs) -> R {
        // Automatically handles mixed-type expressions
    }
}
```

### Phase 2: StaticContext Simplification

**Goal**: Remove complex scope syntax, make it look like `DynamicContext`

```rust
impl StaticContext {
    /// Same interface as DynamicContext but compile-time optimized
    pub fn var<T: ExprType>(&mut self) -> Var<T> {
        // Compile-time variable registration
    }
    
    /// Zero-overhead evaluation
    pub fn eval<R>(&self, expr: &Expr<R>, inputs: &Inputs) -> R {
        // Compile-time specialized evaluation
    }
}
```

### Phase 3: Expression Unification

**Goal**: Same expression types work with both contexts

```rust
// Common expression types that work with both contexts
pub struct Expr<T> {
    // Internal representation abstracts over static/dynamic
}

impl<T> Expr<T> {
    // Operators work the same regardless of context
    pub fn add<U>(self, other: Expr<U>) -> Expr<T::Output>
    where T: Add<U> { ... }
}
```

## Benefits

### For Users

1. **Simple mental model**: Static vs Dynamic, not hetero vs non-hetero
2. **Consistent API**: Same methods, same syntax, same patterns
3. **No choice paralysis**: Use Static for performance, Dynamic for flexibility
4. **Zero learning curve**: Master one, automatically know the other

### For Implementation

1. **Code reuse**: Shared trait implementations
2. **Consistent optimization**: Same algorithms work for both
3. **Better testing**: Test once, works for both
4. **Future-proof**: Easy to add new features to both simultaneously

## Migration Path

### Step 1: Enhance DynamicContext
- Add native heterogeneous support
- Implement array indexing operators  
- Create unified `Inputs` type

### Step 2: Simplify StaticContext
- Remove complex scope builders
- Add simple `var<T>()` method
- Implement same evaluation interface

### Step 3: Unify Expression Types
- Create common `Expr<T>` type
- Implement shared operators
- Add automatic type promotion

### Step 4: Deprecate Old APIs
- Phase out `HeteroContext`
- Remove scope builders
- Provide migration guides

## Success Criteria

1. **API Parity**: Both contexts support identical operations
2. **Zero Overhead**: Static context maintains current performance
3. **User Simplicity**: Single interface to learn
4. **Implementation Elegance**: Shared code, minimal duplication

## Example: Before and After

### Before (Current State)
```rust
// Users must choose between multiple APIs
let dynamic = DynamicContext::new();  // f64 only
let hetero = HeteroContext::new();    // Mixed types, complex syntax
let static_ctx = Context::new_f64();  // Scopes, complex builder pattern

// Different syntaxes for same operations
let x1 = dynamic.var();               // Dynamic: implicit f64
let x2 = hetero.var::<f64>();         // Hetero: explicit type, complex eval
let x3 = static_ctx.new_scope(|scope| { // Static: scope builders
    let (x, _) = scope.auto_var();
    x
});
```

### After (Unified Design)
```rust
// Users choose based on performance characteristics only
let static_ctx = StaticContext::new();   // Compile-time optimized
let dynamic_ctx = DynamicContext::new(); // Runtime flexible

// Identical API for both
let x1 = static_ctx.var::<f64>();     // Same syntax
let x2 = dynamic_ctx.var::<f64>();    // Same syntax
let array1 = static_ctx.var::<Vec<f64>>();  // Hetero support built-in
let array2 = dynamic_ctx.var::<Vec<f64>>(); // Hetero support built-in

// Same evaluation interface
let result1 = static_ctx.eval(&expr, &inputs);  // Zero overhead
let result2 = dynamic_ctx.eval(&expr, &inputs); // Runtime flexible
```

The unified design eliminates cognitive overhead while maintaining all current capabilities. Users focus on their performance needs (static vs dynamic) rather than implementation details (hetero vs non-hetero). 