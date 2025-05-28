# Developer Notes: Expression Types and Architecture

## Overview

The mathjit codebase implements a sophisticated mathematical expression system using the **Final Tagless** approach to solve the expression problem. This document explains the different AST and expression types, their roles, and how they work together.

## Core Architecture: Final Tagless Approach

The final tagless approach uses traits with Generic Associated Types (GATs) to represent mathematical operations. This enables:

1. **Easy extension of operations** without modifying existing code
2. **Easy addition of new interpreters** without modifying existing operations  
3. **Zero intermediate representation** - expressions compile directly to target representations

## Expression Types Hierarchy

### 1. Core Trait: `MathExpr`

**Location**: `src/final_tagless.rs`

The foundation trait that defines mathematical operations using Generic Associated Types:

```rust
pub trait MathExpr {
    type Repr<T>;  // The representation type parameterized by value type
    
    fn constant<T: NumericType>(value: T) -> Self::Repr<T>;
    fn var<T: NumericType>(name: &str) -> Self::Repr<T>;
    fn add<L, R, Output>(...) -> Self::Repr<Output>;
    // ... other operations
}
```

**Role**: Defines the interface for all mathematical interpreters. Any type implementing this trait can interpret mathematical expressions.

### 2. Extension Traits

#### `StatisticalExpr`
**Location**: `src/final_tagless.rs`

Extends `MathExpr` with statistical functions:
- `logistic(x)` - Logistic/sigmoid function
- `softplus(x)` - Softplus function  
- `sigmoid(x)` - Alias for logistic

**Role**: Demonstrates how to extend the system with new operations without modifying existing code.

#### `SummationExpr` 
**Location**: `src/final_tagless.rs` (trait definition), `src/summation.rs` (implementation)

Extends `MathExpr` with summation operations:
- `sum_finite()` - Finite summations
- `sum_infinite()` - Infinite summations
- `sum_telescoping()` - Telescoping sums

**Role**: Adds symbolic summation capabilities with algebraic manipulation.

### 3. Concrete Representations

#### `ASTRepr<T>` - Abstract Syntax Tree
**Location**: `src/final_tagless.rs`

An enum representing mathematical expressions as a tree structure:

```rust
pub enum ASTRepr<T> {
    Constant(T),
    Variable(usize),  // Uses indices for performance
    Add(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    Mul(Box<ASTRepr<T>>, Box<ASTRepr<T>>),
    // ... other operations
}
```

**Key Features**:
- **Index-based variables** for performance (not string names)
- **Generic over numeric types** (f64, f32, etc.)
- **Operator overloading** support for natural syntax
- **Serializable** and suitable for JIT compilation

**Role**: The primary intermediate representation for expressions that need to be analyzed, optimized, or compiled.

### 4. Interpreters (Implementations of `MathExpr`)

#### `DirectEval` - Immediate Evaluation
**Location**: `src/final_tagless.rs`

```rust
impl MathExpr for DirectEval {
    type Repr<T> = T;  // Direct values, no intermediate representation
}
```

**Role**: 
- Immediate evaluation using native Rust operations
- Zero overhead - direct mapping to native operations
- Reference implementation for testing correctness
- Used for constant folding and optimization

#### `PrettyPrint` - String Generation  
**Location**: `src/final_tagless.rs`

```rust
impl MathExpr for PrettyPrint {
    type Repr<T> = String;  // String representation
}
```

**Role**:
- Generate human-readable mathematical notation
- Debugging and documentation
- Expression visualization
- LaTeX/MathML generation (future)

#### `ASTEval` - AST Construction
**Location**: `src/final_tagless.rs`

```rust
impl MathExpr for ASTEval {
    type Repr<T> = ASTRepr<T>;  // Builds AST trees
}
```

**Role**:
- Constructs `ASTRepr` trees for later compilation
- Bridge between final tagless and traditional AST approaches
- Used by JIT compilation pipeline

### 5. Specialized Traits for JIT Compilation

#### `ASTMathExpr` - Homogeneous f64 JIT
**Location**: `src/final_tagless.rs`

Simplified trait that works only with f64 for practical JIT compilation:

```rust
pub trait ASTMathExpr {
    type Repr;  // Always ASTRepr<f64>
    fn constant(value: f64) -> Self::Repr;
    fn var(index: usize) -> Self::Repr;
    // ... operations
}
```

**Role**: Practical compromise for JIT compilation where type homogeneity is needed.

#### `ASTMathExprf64` - Explicit f64 JIT
**Location**: `src/final_tagless.rs`

Even more explicit f64-only trait for JIT compilation.

**Role**: Ensures type safety and performance for JIT compilation backends.

### 6. Ergonomic Wrappers

#### `Expr<E, T>` - Operator Overloading Wrapper
**Location**: `src/lib.rs` (expr module)

```rust
pub struct Expr<E: MathExpr, T> {
    repr: E::Repr<T>,
    _phantom: PhantomData<E>,
}
```

**Role**:
- Enables natural operator syntax: `x + y * z`
- Type-safe wrapper around final tagless representations
- Bridges between final tagless and traditional OOP approaches

#### `MathBuilder` - High-Level API
**Location**: `src/ergonomics.rs`

```rust
pub struct MathBuilder {
    builder: ExpressionBuilder,
    optimizer: Option<SymbolicOptimizer>,
    // ...
}
```

**Role**:
- Primary user-facing API
- Manages variable registries
- Integrates optimization and compilation
- Provides mathematical constants and presets

### 7. Supporting Infrastructure

#### `ExpressionBuilder` - Variable Management
**Location**: `src/final_tagless.rs`

```rust
pub struct ExpressionBuilder {
    registry: VariableRegistry,
}
```

**Role**:
- Maps between string variable names and efficient indices
- Manages variable scoping and evaluation
- Provides both named and indexed variable access

#### `VariableRegistry` - Name/Index Mapping
**Location**: `src/final_tagless.rs`

**Role**:
- Thread-safe mapping between variable names and indices
- Enables user-friendly string names while using efficient indices internally
- Supports both global and local variable scopes

## Expression Type Relationships

```
MathExpr (trait)
├── StatisticalExpr (extension trait)
├── SummationExpr (extension trait)
├── DirectEval (immediate evaluation)
├── PrettyPrint (string generation)
└── ASTEval (AST construction)
    └── ASTRepr<T> (concrete AST)
        ├── Used by JIT compilation
        ├── Used by symbolic optimization
        └── Used by automatic differentiation

Ergonomic Layer:
├── Expr<E, T> (operator overloading)
├── MathBuilder (high-level API)
└── ExpressionBuilder (variable management)
```

## Usage Patterns

### 1. Direct Mathematical Computation
```rust
// Using DirectEval for immediate computation
fn quadratic<E: MathExpr>(x: E::Repr<f64>) -> E::Repr<f64> {
    E::add(E::mul(E::constant(2.0), E::pow(x.clone(), E::constant(2.0))), E::constant(1.0))
}

let result = quadratic::<DirectEval>(DirectEval::var("x", 3.0));
// result = 19.0 (2*9 + 1)
```

### 2. Expression Building for Compilation
```rust
// Using ASTEval to build expressions for JIT compilation
let expr = quadratic::<ASTEval>(ASTEval::var(0)); // Variable at index 0
// expr is now an ASTRepr<f64> ready for compilation
```

### 3. Pretty Printing for Debugging
```rust
// Using PrettyPrint for human-readable output
let pretty = quadratic::<PrettyPrint>(PrettyPrint::var("x"));
// pretty = "((2 * (x ^ 2)) + 1)"
```

### 4. Ergonomic API Usage
```rust
// Using MathBuilder for natural syntax
let mut math = MathBuilder::new();
let x = math.var("x");
let expr = &math.constant(2.0) * &x.pow_ref(&math.constant(2.0)) + &math.constant(1.0);
let result = math.eval(&expr, &[("x", 3.0)]);
```

## Design Benefits

### 1. Expression Problem Solution
- **Add new operations**: Create extension traits like `StatisticalExpr`
- **Add new interpreters**: Implement `MathExpr` for new types
- **No modification** of existing code required

### 2. Performance Flexibility
- **Zero-cost abstractions**: `DirectEval` compiles to native operations
- **Efficient compilation**: `ASTRepr` optimized for JIT compilation
- **Index-based variables**: O(1) variable lookup instead of string hashing

### 3. Type Safety
- **Generic over numeric types**: Works with f64, f32, AD types, etc.
- **Compile-time guarantees**: Type system prevents many runtime errors
- **Trait bounds**: Ensure operations are valid for given types

### 4. Extensibility
- **Modular design**: Each component has a single responsibility
- **Plugin architecture**: Easy to add new backends, optimizations, etc.
- **Backward compatibility**: New features don't break existing code

## Common Pitfalls and Solutions

### 1. Variable Index vs Name Confusion
**Problem**: Mixing string-based and index-based variable access.

**Solution**: Use `ExpressionBuilder` or `MathBuilder` for consistent variable management.

### 2. Type Parameter Complexity
**Problem**: Complex generic bounds in function signatures.

**Solution**: Use the `NumericType` helper trait to bundle common bounds.

### 3. Interpreter Selection
**Problem**: Choosing the wrong interpreter for the task.

**Solution**: 
- Use `DirectEval` for immediate computation
- Use `ASTEval` for building expressions to compile
- Use `PrettyPrint` for debugging and display

### 4. Performance vs Flexibility Trade-offs
**Problem**: Generic code can be slower than specialized code.

**Solution**: The design allows both - use generics for flexibility, specialize for performance-critical paths.

## Future Extensions

The architecture is designed to support:

1. **New numeric types**: Complex numbers, arbitrary precision, etc.
2. **New operations**: Matrix operations, special functions, etc.  
3. **New interpreters**: GPU compilation, symbolic computation, etc.
4. **New optimizations**: Algebraic simplification, numerical stability, etc.

## Testing Strategy

Each expression type should be tested with:

1. **Correctness tests**: Compare `DirectEval` results with known values
2. **Consistency tests**: Ensure all interpreters produce equivalent results
3. **Performance tests**: Benchmark critical paths
4. **Property tests**: Use proptest for algebraic properties 