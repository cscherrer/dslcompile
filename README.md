# MathJIT

High-performance symbolic mathematics with final tagless design, egglog optimization, and **Rust hot-loading compilation**.

## Features

- **Final Tagless Approach**: Type-safe expression building with multiple interpreters
- **Symbolic Optimization**: Algebraic simplification using egglog equality saturation
- **Rust Hot-Loading**: Primary compilation backend for maximum performance
- **Cranelift JIT**: Optional fast JIT compilation for rapid iteration
- **Symbolic Automatic Differentiation**: Compute derivatives with shared subexpressions
- **Multiple Backends**: Choose the best compilation strategy for your use case

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Final Tagless Layer                     │
│  (Expression Building & Type Safety)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Symbolic Optimization                       │
│  (Algebraic Simplification & Rewrite Rules)                │
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

## Quick Start

### Basic Usage

```rust
use mathjit::prelude::*;

// Build expressions using the final tagless approach
let expr = JITEval::add(
    JITEval::mul(JITEval::var("x"), JITEval::constant(2.0)),
    JITEval::constant(1.0)
); // 2*x + 1

// Optimize symbolically
let mut optimizer = SymbolicOptimizer::new()?;
let optimized = optimizer.optimize(&expr)?;

// Generate and compile Rust code
let codegen = RustCodeGenerator::new();
let rust_code = codegen.generate_function(&optimized, "my_function")?;

let compiler = RustCompiler::with_opt_level(RustOptLevel::O2);
// ... compile and load dynamic library
```

### Symbolic Automatic Differentiation

```rust
use mathjit::symbolic_ad::*;

// Create a function: f(x) = x² + 2x + 1
let f = convenience::quadratic(1.0, 2.0, 1.0);

// Compute function and derivative together with shared subexpressions
let mut ad = SymbolicAD::new()?;
let result = ad.compute_with_derivatives(&f)?;

println!("f(x) = {:?}", result.function);
println!("f'(x) = {:?}", result.first_derivatives["x"]);
println!("Optimization ratio: {:.2}", result.stats.optimization_ratio());
```

## Compilation Strategies

### 1. Rust Hot-Loading (Primary)

Best for: Complex expressions, maximum performance, production use

```rust
let strategy = CompilationStrategy::HotLoadRust {
    source_dir: PathBuf::from("./generated"),
    lib_dir: PathBuf::from("./libs"),
    opt_level: RustOptLevel::O2,
};
```

### 2. Cranelift JIT (Optional)

Best for: Simple expressions, rapid iteration, low latency

```rust
// Enable with: cargo build --features cranelift
let strategy = CompilationStrategy::CraneliftJIT;
```

### 3. Adaptive Strategy

Automatically chooses the best backend based on expression characteristics:

```rust
let strategy = CompilationStrategy::Adaptive {
    call_threshold: 100,
    complexity_threshold: 25,
};
```

## Features

- `default`: Includes symbolic optimization with egglog
- `optimization`: Symbolic optimization with egglog (same as default)
- `cranelift`: Optional Cranelift JIT compilation backend
- `all`: All features enabled

## Performance

MathJIT's Rust hot-loading backend generates highly optimized native code that can outperform traditional interpreters by orders of magnitude:

- **Symbolic optimization**: Reduces expression complexity before compilation
- **Subexpression sharing**: Eliminates redundant computations in derivatives
- **Native compilation**: Full Rust compiler optimizations
- **Hot-loading**: Runtime compilation and loading of optimized code

## Examples

See the `examples/` directory for comprehensive examples including:

- Real-world automatic differentiation performance comparisons
- Symbolic optimization showcases
- Backend comparison benchmarks

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option. 