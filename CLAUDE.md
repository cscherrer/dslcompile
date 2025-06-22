# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- `cargo test` - Run all tests
- `cargo test --package dslcompile` - Test main package only
- `cargo test test_name` - Run specific test
- `cargo test --features all` - Test with all features enabled
- `cargo test --features all-no-llvm` - Test all features except LLVM JIT
- `cargo test --features llvm_jit` - Test with LLVM JIT backend (requires LLVM 18)

### Building & Checking
- `cargo build` - Build the project
- `cargo build --release` - Release build with optimizations
- `cargo build --features llvm_jit` - Build with LLVM JIT backend (requires LLVM 18)
- `cargo check` - Fast syntax/type checking
- `cargo clippy` - Lint checking

### Benchmarking
- `cargo bench` - Run performance benchmarks (uses Divan)
- `cargo run --example expression_optimization --release` - Optimization benchmarks

### Examples
- `cargo run --example multiset_demonstration` - Multiset functionality demo
- `cargo run --example simple_math_demo` - Basic mathematical operations demo
- `cargo run --example measures_library_demo` - Measures library patterns (in root `examples/`)
- `cargo run --example llvm_jit_demo --features llvm_jit` - LLVM JIT compilation demo
- `cargo run --example llvm_optimization_analysis --features llvm_jit` - LLVM optimization analysis
- `cargo run --example llvm_random_data_benchmark --features llvm_jit` - LLVM performance benchmarks
- Examples are in `dslcompile/examples/`

## Architecture Overview

DSLCompile is a mathematical expression compiler with a three-layer optimization strategy:

### 1. Expression Building Layer (Final Tagless)
- **Core trait**: `MathExpr` with Generic Associated Types (GATs)
- **Two primary contexts**:
  - `StaticContext`: Compile-time optimization, zero-overhead, HList support
  - `DynamicContext`: Runtime flexibility, JIT compilation, symbolic optimization
- **Key types**: `ASTRepr<T>`, `DynamicExpr`, `StaticExpr`

### 2. Symbolic Optimization Layer
- **Engine**: Uses egg for algebraic simplification and dependency analysis
- **Location**: `src/symbolic/` module
- **Entry point**: `egg_optimizer::optimize_simple_sum_splitting` function
- **Features**: Sum splitting, coefficient factoring, dependency tracking
- **Status**: Currently under development with compilation issues

### 3. Compilation Backend Layer
- **Rust Backend**: Hot-loading compilation (`src/backends/rust_codegen.rs`) - Primary backend
- **LLVM JIT Backend**: Direct JIT compilation (`src/backends/llvm_jit.rs`) - Maximum performance
- **Static Backend**: Zero-overhead inline compilation (`src/backends/static_compiler.rs`)
- **Output**: Native performance compiled functions

## Key Design Patterns

### Two-Context Architecture
After recent consolidation, there are exactly two clean interfaces:

1. **StaticContext** (`src/contexts/static_context/`):
   - Zero-overhead, compile-time scoped variables
   - HList heterogeneous type support
   - Automatic scope management

2. **DynamicContext** (`src/contexts/dynamic/`):
   - Runtime flexibility with variable registries
   - Operator overloading syntax
   - Integration with symbolic optimization

### Variable Management
- **Internal**: Uses integer indices for performance
- **External**: String names for user convenience
- **Registry**: `VariableRegistry` maps names ↔ indices thread-safely


## Module Structure

### Core Modules
- `src/ast/`: Abstract syntax tree representations and evaluation
- `src/contexts/`: StaticContext and DynamicContext implementations
- `src/symbolic/`: Symbolic optimization using egg
- `src/backends/`: Code generation and compilation
- `src/error.rs`: Unified error handling

### Important Files
- `src/lib.rs`: Main API exports and prelude
- `src/contexts/dynamic/expression_builder.rs`: DynamicContext implementation
- `src/contexts/static_context/static_scoped.rs`: StaticContext core
- `src/symbolic/egg_optimizer.rs`: Egg-based symbolic optimization
- `src/symbolic/symbolic.rs`: SymbolicOptimizer implementation
- `src/backends/rust_codegen.rs`: Rust code generation

## Workspace Structure

This is a Cargo workspace with two packages:
- `dslcompile/`: Main library package
- `dslcompile-macros/`: Procedural macros for compile-time optimization

## Features

- `default = ["optimization"]`: Includes egg symbolic optimization
- `optimization`: Enables symbolic optimization with egg
- `ad_trait`: Automatic differentiation trait support
- `llvm_jit`: Enables LLVM JIT compilation backend (requires LLVM 18)
- `all`: All features enabled (includes LLVM JIT)
- `all-no-llvm`: All features except LLVM JIT (for environments without LLVM 18)

## Testing Strategy

- **Unit tests**: In `src/` modules and `tests/` directory
- **Property tests**: Uses `proptest` for algebraic property verification
- **Integration tests**: End-to-end pipeline testing
- **Benchmarks**: Performance regression testing with Divan

## Key Memories

### Egg Optimization (Current)
- Dependency analysis tracks free variables in expressions for safe coefficient factoring
- Sum splitting: `Σ(a*x + b*x)` → `(a+b)*Σ(x)` when coefficients are independent
- Implementation uses egg e-graph optimization library
- **Status**: Under active development, some compilation issues remain
- Examples: `test_dependency_analysis.rs`, `simple_egg_demo.rs` (may need updates)

### Migration Notes (Historical)
- Uses egg library for e-graph based optimization
- Previous experiments with egglog informed current design
- Current implementation focuses on reliability and simplicity

[Rest of the file remains unchanged...]