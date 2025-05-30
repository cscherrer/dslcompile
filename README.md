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