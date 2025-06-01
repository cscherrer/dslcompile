# MathCompile

**Symbolic mathematics compiler for Rust**

Transform symbolic mathematical expressions into optimized native code with automatic differentiation support.

## Overview

MathCompile provides a compilation pipeline for mathematical expressions:

- **Symbolic optimization** using algebraic simplification before compilation
- **Native code generation** through Rust's compiler or optional Cranelift JIT
- **Automatic differentiation** with shared subexpression optimization
- **Final tagless design** for type-safe, extensible expression building

## Core Capabilities

### Beautiful Mathematical Syntax

MathCompile provides intuitive syntax for building complex mathematical expressions and can attempt to discover non-trivial mathematical simplifications automatically:

```rust
use mathcompile::prelude::*;

// Challenge: Discover hidden simplifications in complex nested expressions
let math = MathBuilder::new();
let x = math.var("x");
let y = math.var("y");
let z = math.var("z");
let a = math.var("a");
let b = math.var("b");

// Build a complex nested expression: ln(e^x * e^y * e^z) + ln(e^a) - ln(e^b)
// Hidden pattern: This should simplify to x + y + z + a - b (not obvious!)
let exp_x = x.clone().exp();
let exp_y = y.clone().exp();
let exp_z = z.clone().exp();
let exp_a = a.clone().exp();
let exp_b = b.clone().exp();

let product = &exp_x * &exp_y * &exp_z;
let complex_expr = product.ln() + exp_a.ln() - exp_b.ln();

// Attempt automatic mathematical discovery
#[cfg(feature = "optimization")]
{
    use mathcompile::symbolic::native_egglog::NativeEgglogOptimizer;
    
    let mut optimizer = NativeEgglogOptimizer::new()?;
    let optimized = optimizer.optimize(&complex_expr.as_ast())?;
    
    // The system attempts to discover the hidden pattern using:
    // - ln(e^x) = x simplification rules
    // - ln(a*b) = ln(a) + ln(b) product rules  
    // - e^x * e^y = e^(x+y) exponential rules
    
    println!("Original: 12 operations, depth 7");
    println!("Discovered: {} operations, depth {}", 
        count_operations(&optimized), 
        expression_depth(&optimized)
    );
}

// Verify mathematical correctness
let result = math.eval(&complex_expr, &[
    ("x", 2.0), ("y", 3.0), ("z", 1.0), ("a", 4.0), ("b", 0.5)
]);
let expected_simple = 2.0 + 3.0 + 1.0 + 4.0 - 0.5; // x + y + z + a - b
assert_eq!(result, expected_simple); // Both equal 9.5

// Performance shows the value of discovery: simple form is 1.15x faster!
```

### Expression Building and Optimization
```rust
use mathcompile::prelude::*;

// Define symbolic expression
let mut math = MathBuilder::new();
let x = math.var("x");
let expr = math.poly(&[1.0, 2.0, 3.0], &x); // 1 + 2x + 3x² (coefficients in ascending order)

// Algebraic simplification
let optimized = math.optimize(&expr)?;

// Direct evaluation (fastest for immediate use)
let result = DirectEval::eval_with_vars(&optimized, &[3.0]); // x = 3.0
assert_eq!(result, 34.0); // 1 + 2*3 + 3*9 = 34

// Generate Rust code for compilation
let codegen = RustCodeGenerator::new();
let rust_code = codegen.generate_function(&optimized, "my_function")?;

// Compile and load the function
let compiler = RustCompiler::new();
let compiled_func = compiler.compile_and_load(&rust_code, "my_function")?;
let compiled_result = compiled_func.call(3.0)?;
assert_eq!(compiled_result, 34.0);
```

### Code Generation Comparison

The factorization demo shows how different mathematical representations generate different code:

**Expanded Form** (complex, many operations):
```rust
pub extern "C" fn expanded_form(var_0: f64, var_1: f64) -> f64 {
    return (((((((({ let temp2 = var_0 * var_0; let temp4 = temp2 * temp2; temp4 * temp4 } 
    + ((8_f64 * var_0.powi(7)) * var_1)) + ((28_f64 * { let temp = var_0 * var_0 * var_0; temp * temp }) * var_1 * var_1))
    // ... many more terms
}
```

**Factored Form** (elegant, efficient):
```rust
pub extern "C" fn factored_form(var_0: f64, var_1: f64) -> f64 {
    return { let temp2 = (var_0 + var_1) * (var_0 + var_1); let temp4 = temp2 * temp2; temp4 * temp4 };
}
```

**Performance Result**: Factored form is **1.77x faster** than expanded form!

### Automatic Differentiation
```rust
// Define function using MathBuilder
let mut math = MathBuilder::new();
let x = math.var("x");
let f = math.poly(&[1.0, 2.0, 1.0], &x); // 1 + 2x + x²

// Convert to optimized AST
let optimized_f = math.optimize(&f)?;

// Compute function and derivatives
let mut ad = SymbolicAD::new()?;
let result = ad.compute_with_derivatives(&optimized_f)?;

println!("f(x) = 1 + 2x + x²");
println!("f'(x) = 2 + 2x (computed symbolically)");
println!("Shared subexpressions: {}", result.stats.shared_subexpressions_count);
```

### Multiple Compilation Backends
```rust
// Rust code generation (primary backend)
let codegen = RustCodeGenerator::new();
let rust_code = codegen.generate_function(&optimized, "my_func")?;

let compiler = RustCompiler::new();
let compiled_func = compiler.compile_and_load(&rust_code, "my_func")?;
let result = compiled_func.call(3.0)?;

// Cranelift JIT (optional, requires "cranelift" feature)
#[cfg(feature = "cranelift")]
{
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_single_var(&optimized, "x")?;
    let jit_result = jit_func.call_single(3.0);
}
```

## Installation

Add to your `Cargo.toml`:
```toml
[dependencies]
mathcompile = "0.1"

# Optional: Enable Cranelift JIT backend
# mathcompile = { version = "0.1", features = ["cranelift"] }
```

## Basic Usage

```rust
use mathcompile::prelude::*;

// Create mathematical expressions
let mut math = MathBuilder::new();
let x = math.var("x");
let expr = math.add(
    &math.add(&math.mul(&x, &x), &math.mul(&math.constant(2.0), &x)),
    &math.constant(1.0)
); // x² + 2x + 1

// Optimize symbolically
let optimized = math.optimize(&expr)?;

// Evaluate efficiently
let result = DirectEval::eval_with_vars(&optimized, &[3.0]); // x = 3.0
assert_eq!(result, 16.0); // 9 + 6 + 1

// Generate and compile Rust code
let codegen = RustCodeGenerator::new();
let rust_code = codegen.generate_function(&optimized, "quadratic")?;

let compiler = RustCompiler::new();
let compiled_func = compiler.compile_and_load(&rust_code, "quadratic")?;
let compiled_result = compiled_func.call(3.0)?;
assert_eq!(compiled_result, 16.0);

// JIT compilation (if cranelift feature enabled)
#[cfg(feature = "cranelift")]
{
    let compiler = JITCompiler::new()?;
    let compiled = compiler.compile_single_var(&optimized, "x")?;
    let jit_result = compiled.call_single(3.0);
    assert_eq!(jit_result, 16.0);
}
```

## Documentation

- **[Developer Notes](DEVELOPER_NOTES.md)** - Architecture overview and expression types
- **[Roadmap](ROADMAP.md)** - Project status and planned features  
- **[Examples](examples/)** - Usage examples and demonstrations
- **[API Documentation](https://docs.rs/mathcompile)** - Complete API reference

## Architecture

MathCompile uses a **final tagless** approach to solve the expression problem:

- **Extensible operations** - Add new mathematical functions without modifying existing code
- **Multiple interpreters** - Same expressions work with evaluation, optimization, and compilation
- **Type safety** - Compile-time guarantees for mathematical operations

```text
┌─────────────────────────────────────────────────────────────┐
│                    Expression Building                       │
│  (Final Tagless Design + MathBuilder API)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Symbolic Optimization                       │
│  (Algebraic Simplification + Egglog Integration)            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Compilation Backends                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    Rust     │  │  Cranelift  │  │  Future Backends    │  │
│  │ Hot-Loading │  │     JIT     │  │   (LLVM, etc.)      │  │
│  │ (Primary)   │  │ (Optional)  │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Final Tagless Design**: Type-safe expression building with multiple interpreters
- **Symbolic Optimization**: Advanced algebraic simplification using egglog
- **Multiple Backends**: Rust hot-loading (primary) and optional Cranelift JIT
- **Automatic Differentiation**: Forward and reverse mode with symbolic optimization
- **Advanced Summation**: Multi-dimensional sums with convergence analysis
- **Domain Analysis**: Abstract interpretation ensuring mathematical transformations are only applied when valid
- **A-Normal Form**: Intermediate representation with scope-aware common subexpression elimination

## Technical Notes

- **Polynomial coefficients**: The `poly` function takes coefficients in ascending order of powers