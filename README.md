# MathJIT

**High-performance symbolic mathematics compiler for Rust**

Transform symbolic mathematical expressions into highly optimized native code with automatic differentiation support.

## Why MathJIT?

When mathematical computation is expensive enough to warrant compilation overhead, MathJIT delivers:

- **Symbolic optimization** before compilation eliminates redundant operations
- **Native code generation** through Rust's compiler for maximum performance  
- **Automatic differentiation** with shared subexpression optimization
- **JIT compilation** for rapid iteration during development
- **Production-ready** hot-loading for deployment scenarios

Perfect for researchers, quantitative analysts, and engineers working with complex mathematical models where computation time matters.

## Key Capabilities

### 🔬 **Symbolic → Numeric Optimization**
```rust
// Define symbolic expression
let mut math = MathBuilder::new();
let x = math.var("x");
let expr = math.poly(&[1.0, 2.0, 3.0], &x); // 1 + 2x + 3x² (coefficients in ascending order)

// Automatic algebraic simplification
let optimized = math.optimize(&expr)?;

// Evaluate efficiently with indexed variables (fastest for immediate use)
let result = DirectEval::eval_with_vars(&optimized, &[3.0]); // x = 3.0

// Or generate optimized Rust code for maximum performance
let codegen = RustCodeGenerator::new();
let rust_code = codegen.generate_function(&optimized, "my_function")?;

// Compile and load the function (paths auto-generated from function name)
let compiler = RustCompiler::new();
let compiled_func = compiler.compile_and_load(&rust_code, "my_function")?;
let compiled_result = compiled_func.call(3.0)?; // Blazing fast native execution!
```

### 📈 **Automatic Differentiation**
```rust
// Define a complex function using MathBuilder first
let mut math = MathBuilder::new();
let x = math.var("x");
let f = math.poly(&[1.0, 2.0, 1.0], &x); // 1 + 2x + x² (coefficients in ascending order)

// Convert to optimized AST
let optimized_f = math.optimize(&f)?;

// Compute function and derivatives with optimization
let mut ad = SymbolicAD::new()?;
let result = ad.compute_with_derivatives(&optimized_f)?;

println!("f(x) = polynomial (1 + 2x + x²)");
println!("f'(x) computed (derivative of 1 + 2x + x² = 2 + 2x)");
println!("Shared subexpressions: {}", result.stats.shared_subexpressions_count);
```

### ⚡ **Multiple Compilation Backends**
```rust
// Cranelift JIT for rapid iteration (if feature enabled)
#[cfg(feature = "cranelift")]
{
    let compiler = JITCompiler::new()?;
    let jit_func = compiler.compile_single_var(&optimized, "x")?;
    let fast_result = jit_func.call_single(3.0);
}

// Rust code generation for maximum performance
let codegen = RustCodeGenerator::new();
let rust_code = codegen.generate_function(&optimized, "my_func")?;

// Compile and load with auto-generated paths
let compiler = RustCompiler::new();
let compiled_func = compiler.compile_and_load(&rust_code, "my_func")?;
let compiled_result = compiled_func.call(3.0)?;
```

## Quick Start

Add to your `Cargo.toml`:
```toml
[dependencies]
mathjit = "0.1"

# Optional: Enable Cranelift JIT backend
# mathjit = { version = "0.1", features = ["cranelift"] }
```

### Basic Usage

```rust
use mathjit::prelude::*;

// Create mathematical expressions
let mut math = MathBuilder::new();
let x = math.var("x");
let expr = math.add(&math.add(&math.mul(&x, &x), &math.mul(&math.constant(2.0), &x)), &math.constant(1.0)); // x² + 2x + 1

// Optimize symbolically
let optimized = math.optimize(&expr)?;

// Evaluate efficiently (fastest method)
let result = DirectEval::eval_with_vars(&optimized, &[3.0]); // x = 3.0
assert_eq!(result, 16.0); // 9 + 6 + 1

// Generate and compile Rust code for maximum performance
let codegen = RustCodeGenerator::new();
let rust_code = codegen.generate_function(&optimized, "quadratic")?;

let compiler = RustCompiler::new();
let compiled_func = compiler.compile_and_load(&rust_code, "quadratic")?;
let compiled_result = compiled_func.call(3.0)?; // Native speed execution
assert_eq!(compiled_result, 16.0);

// Or use JIT compilation for rapid iteration (if available)
#[cfg(feature = "cranelift")]
{
    let compiler = JITCompiler::new()?;
    let compiled = compiler.compile_single_var(&optimized, "x")?;
    let fast_result = compiled.call_single(3.0);
    assert_eq!(fast_result, 16.0);
}
```

## Documentation

- **[Developer Notes](DEVELOPER_NOTES.md)** - Architecture overview and expression types
- **[Roadmap](ROADMAP.md)** - Project status and planned features  
- **[Examples](examples/)** - Comprehensive usage examples and benchmarks
- **[API Documentation](https://docs.rs/mathjit)** - Complete API reference

## Architecture

MathJIT uses a **final tagless** approach to solve the expression problem, enabling:

- **Extensible operations** - Add new mathematical functions without modifying existing code
- **Multiple interpreters** - Same expressions work with evaluation, optimization, and compilation
- **Type safety** - Compile-time guarantees for mathematical operations
- **Zero-cost abstractions** - No runtime overhead for expression building

```text
┌─────────────────────────────────────────────────────────────┐
│                    Expression Building                       │
│  (Final Tagless Design + Ergonomic API)                     │
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
│  │ Hot-Loading │  │     JIT     │  │   (LLVM, GPU)       │  │
│  │ (Primary)   │  │ (Optional)  │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **`default`** - Core functionality with symbolic optimization
- **`cranelift`** - Enable Cranelift JIT compilation backend  
- **`all`** - All available features

## Use Cases

- **Scientific Computing** - Optimize complex mathematical models
- **Quantitative Finance** - High-frequency trading algorithms  
- **Machine Learning** - Custom loss functions and optimizers
- **Engineering Simulation** - Physics-based modeling
- **Research** - Rapid prototyping of mathematical algorithms

## Contributing

We welcome contributions! Please see our [Developer Notes](DEVELOPER_NOTES.md) for architecture details and [Roadmap](ROADMAP.md) for planned features.

## License

Licensed under the MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT). 