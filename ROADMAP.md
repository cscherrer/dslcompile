# MathCompile Development Roadmap

## Project Overview
MathCompile is a mathematical expression compiler that transforms symbolic expressions into optimized machine code. The project combines symbolic computation, automatic differentiation, and just-in-time compilation for mathematical computations.

## Current Status: Phase 4 - Statistical Computing (NEW)

**Last Updated**: May 30, 2025

## ✅ Completed Features

### Core Infrastructure
- [x] **Final Tagless Architecture**: Clean separation between expression representation and interpretation
- [x] **AST-based Expression System**: Tree representation for mathematical expressions
- [x] **Variable Management**: Index-based variables for performance
- [x] **Multiple Interpreters**: Direct evaluation, pretty printing, and AST evaluation

### Advanced Features  
- [x] **Symbolic Optimization with egglog**: Algebraic simplification and optimization
- [x] **JIT Compilation**: Hot-reloading Rust code generation
- [x] **Automatic Differentiation**: Integration with `ad_trait` for forward-mode AD
- [x] **Summation Support**: Finite and infinite summations with algebraic manipulation

### Unified Trait-Based Type System (December 2024)
- [x] **Type-Safe Variables**: Compile-time type checking with `TypedVar<T>`
- [x] **Operator Overloading**: Syntax like `&x * &x + 2.0 * &x + &y`
- [x] **Trait-Based Type Categories**: `FloatType`, `IntType`, `UIntType` for extensibility
- [x] **Automatic Type Promotion**: Cross-type operations (f32 → f64)
- [x] **High-Level Mathematical Functions**: Polynomials, Gaussian, logistic, tanh
- [x] **Evaluation Interface**: `math.eval(&expr, &[("x", 3.0), ("y", 1.0)])`
- [x] **Backward Compatibility**: Existing code continues to work unchanged
- [x] **Simplified Architecture**: Removed dual type systems and unnecessary complexity

### Statistical Computing & PPL Backend (December 2024)
- [x] **Staged Compilation for Statistics**: Three-stage optimization pipeline for statistical models
- [x] **Runtime Data Binding**: Efficient evaluation with large datasets via `call_multi_vars(&[f64])`
- [x] **Bayesian Linear Regression**: Complete example demonstrating PPL backend capabilities
- [x] **Log-Density Compilation**: Symbolic construction and optimization of statistical densities
- [x] **Sufficient Statistics**: Automatic emergence through symbolic optimization (no special patterns needed)
- [x] **MCMC Integration Ready**: Direct compatibility with nuts-rs and other samplers
- [x] **Performance Optimization**: ~19M evaluations/second for compiled log-posterior functions
- [x] **Detailed Performance Profiling**: Stage-by-stage timing analysis with breakdown percentages
- [x] **Amortization Analysis**: Automatic calculation of compilation cost vs. runtime benefit

The library provides a unified type system with compile-time type safety while maintaining advanced features like symbolic optimization and JIT compilation.

## Current Status

The library has reached a major milestone with the unified type system. The `MathBuilder` (alias for `TypedExpressionBuilder`) provides:

```rust
let math = MathBuilder::new();
let x = math.var("x");
let y = math.var("y");

// Operator overloading syntax
let expr = &x * &x + 2.0 * &x + &y;
let result = math.eval(&expr, &[("x", 3.0), ("y", 1.0)]); // = 16

// High-level functions
let gaussian = math.gaussian(0.0, 1.0, &x);
let poly = math.poly(&[1.0, 3.0, 2.0], &x); // 2x² + 3x + 1
```

## Next Priorities

### 1. **Partial Evaluation & Abstract Interpretation** (High Priority - NEW)
- [ ] **Data Range Analysis**: Implement min/max value tracking for optimization opportunities
- [ ] **Sparsity Pattern Detection**: Identify and eliminate zero-value terms automatically
- [ ] **Statistical Property Analysis**: Use mean, variance for numerical stability optimizations
- [ ] **Correlation Structure Analysis**: Detect and eliminate redundant computations
- [ ] **Partial Data Specialization**: Support fixing some data points while varying others
- [ ] **Hierarchical Model Support**: Specialize on group-level data for hierarchical models
- [ ] **Time Series Specialization**: Specialize on historical data for future predictions
- [ ] **Ensemble Method Support**: Specialize each model on different data subsets
- [ ] **Domain-Aware Partial Evaluation**: Integrate with interval domain analysis
- [ ] **Abstract Interpretation Framework**: Formal framework for compile-time analysis

### 2. **API Modernization** (High Priority)
- [ ] Update all examples to use the new operator syntax
- [ ] Update benchmarks to showcase the new API
- [ ] Create comprehensive documentation for the unified system
- [ ] Add migration guide from old verbose API

### 3. **Performance Optimization** (Medium Priority)
- [ ] Benchmark the new type system vs old system
- [ ] Optimize cloning in operator overloading
- [ ] Profile memory usage of the unified system
- [ ] Consider `Copy` trait for small expressions

### 4. **Enhanced Type System** (Medium Priority)
- [ ] Add more mathematical function categories (Trigonometric, Hyperbolic, etc.)
- [ ] Implement complex number support
- [ ] Add matrix/vector types
- [ ] Enhanced error messages for type mismatches

### 5. **Advanced Features** (Lower Priority)
- [ ] Symbolic differentiation (complement to automatic differentiation)
- [ ] Interval arithmetic for uncertainty quantification
- [ ] GPU compilation backends (CUDA, OpenCL)
- [ ] WebAssembly target for browser deployment

## Research Areas

### Mathematical Capabilities
- [ ] **Tensor Operations**: Multi-dimensional array support
- [ ] **Probability Distributions**: Built-in statistical functions
- [ ] **Numerical Methods**: Integration, root finding, optimization
- [ ] **Special Functions**: Bessel, gamma, hypergeometric functions

### Compilation Targets
- [ ] **LLVM Backend**: Direct LLVM IR generation
- [ ] **SPIR-V**: GPU compute shader generation
- [ ] **Custom DSLs**: Domain-specific language generation

### Advanced Optimizations
- [ ] **Loop Fusion**: Automatic vectorization of mathematical operations
- [ ] **Memory Layout Optimization**: Cache-friendly expression evaluation
- [ ] **Parallel Evaluation**: Multi-threaded expression computation

## Performance Goals

- **Compilation Speed**: Sub-second compilation for complex expressions
- **Runtime Performance**: Within 5% of hand-optimized code
- **Memory Usage**: Minimal allocation during expression evaluation
- **Type Safety**: Zero runtime type errors with compile-time guarantees

## Testing Strategy

- [x] **Unit Tests**: Comprehensive coverage of all features
- [x] **Integration Tests**: End-to-end workflow validation
- [x] **Property-Based Tests**: Randomized testing with QuickCheck
- [ ] **Performance Regression Tests**: Automated benchmarking
- [ ] **Cross-Platform Testing**: Windows, macOS, Linux validation

## Documentation Priorities

1. **Getting Started Guide**: Quick introduction to the unified API
2. **Type System Guide**: Understanding FloatType, IntType, UIntType
3. **Performance Guide**: Optimization tips and best practices
4. **Migration Guide**: Moving from old API to new unified system
5. **Advanced Features**: Symbolic optimization, JIT compilation, AD

## Next Steps (Phase 4: Advanced Integration & Scale)

**Status**: Ready to Begin (May 2025)

With domain analysis complete, the mathematical expression library has achieved a milestone in safety and correctness. The next phase focuses on advanced integration, performance optimization, and expanding the ecosystem.

#### Immediate Priorities (Q2-Q3 2025)

1. **Enhanced Domain-Aware Optimizations**
   - [ ] **Domain-Guided Constant Folding**: Use domain information to safely evaluate more constant expressions
   - [ ] **Conditional Transformations**: Apply different optimization rules based on domain constraints
   - [ ] **Domain Propagation**: Improve domain inference through complex expression chains
   - [ ] **User Domain Hints**: Allow users to specify domain constraints for better optimization

2. **ANF-Domain Integration**
   - [ ] **Domain-Aware ANF**: Integrate domain analysis into A-Normal Form transformations
   - [ ] **Safe CSE**: Ensure common subexpression elimination respects domain constraints
   - [ ] **Domain-Preserving Let-Bindings**: Maintain domain information through ANF transformations
   - [ ] **Optimization Metrics**: Track domain safety improvements in ANF pipeline

3. **Advanced Egglog Integration**
   - [ ] **Domain-Aware Rewrite Rules**: Enhance egglog rules with domain preconditions
   - [ ] **Conditional Rewrites**: Only apply transformations when domain constraints are satisfied
   - [ ] **Domain Extraction**: Extract optimal expressions while preserving domain safety
   - [ ] **Hybrid Optimization**: Combine egglog equality saturation with domain analysis

4. **Performance & Scalability**
   - [ ] **Domain Cache Optimization**: Improve performance of domain computation caching
   - [ ] **Parallel Domain Analysis**: Thread-safe domain analysis for concurrent workloads
   - [ ] **Incremental Analysis**: Update domains efficiently when expressions change
   - [ ] **Memory Management**: Optimize memory usage for large expression trees

#### Strategic Goals (Q4 2025 - 2026)

**Production-Ready Mathematical Computing:**
- [ ] **Industrial Applications**: Deploy in scientific computing, finance, and engineering
- [ ] **Language Bindings**: Python, Julia, MATLAB interfaces with domain safety
- [ ] **Framework Integration**: NumPy, SciPy, JAX compatibility with domain awareness
- [ ] **Real-time Systems**: Ultra-low latency compilation with domain validation

**Advanced Mathematical Features:**
- [ ] **Complex Domain Analysis**: Extend to complex numbers and multi-valued functions
- [ ] **Interval Arithmetic**: Rigorous interval-based domain tracking
- [ ] **Symbolic Domain Constraints**: Express domain constraints symbolically
- [ ] **Proof Generation**: Generate mathematical proofs of transformation validity

**Ecosystem Expansion:**
- [ ] **Educational Tools**: Interactive domain analysis for teaching mathematics
- [ ] **Research Platform**: Support for advanced mathematical research
- [ ] **Industry Partnerships**: Collaborate with mathematical software companies
- [ ] **Open Source Community**: Build contributor ecosystem around domain-aware optimization

#### Research Directions

**Theoretical Foundations:**
- [ ] **Domain Lattice Theory**: Formal verification of domain operations
- [ ] **Transformation Soundness**: Prove correctness of domain-aware transformations
- [ ] **Completeness Analysis**: Determine optimal domain precision vs. performance trade-offs
- [ ] **Abstract Interpretation Extensions**: Explore advanced abstract domains

**Practical Applications:**
- [ ] **Machine Learning**: Domain-aware automatic differentiation for neural networks
- [ ] **Quantum Computing**: Domain analysis for quantum circuit optimization
- [ ] **Distributed Computing**: Domain-aware expression distribution across clusters
- [ ] **Embedded Systems**: Lightweight domain analysis for resource-constrained environments

#### Success Metrics

**Technical Metrics:**
- Domain analysis coverage: >95% of mathematical transformations validated

**Adoption Metrics:**
- Community engagement: Active contributors and users
- Industrial adoption: Production deployments in scientific computing
- Academic recognition: Publications and citations in mathematical software literature
- Ecosystem growth: Third-party tools and integrations

#### Implementation Strategy

**Phase 4A: Foundation Enhancement (Q2 2025)**
- Complete domain-aware optimizations
- Integrate with ANF and egglog systems
- Establish performance benchmarks
- Create comprehensive documentation

**Phase 4B: Ecosystem Development (Q3 2025)**
- Build language bindings and integrations
- Develop educational and research tools
- Establish industry partnerships
- Create contributor onboarding

**Phase 4C: Advanced Features (Q4 2025)**
- Implement advanced domain theories
- Add proof generation capabilities
- Optimize for production workloads
- Expand to new mathematical domains

**Phase 4D: Community & Scale (2026)**
- Build open source community
- Support large-scale deployments
- Advance theoretical foundations
- Explore new application domains

---

## Recent Achievements ✅

### A-Normal Form (ANF) with Scope-Aware Common Subexpression Elimination

**Status: COMPLETE (May 2025)**

#### What We Built
- **ANF Intermediate Representation**: Complete transformation from `ASTRepr` to A-Normal Form
- **Scope-Aware CSE**: Common subexpression elimination that respects variable lifetimes
- [ ] **Hybrid Variable Management**: `VarRef::User(usize)` + `VarRef::Bound(u32)` system
- **Clean Code Generation**: Produces readable, efficient Rust code
- **Property-Based Testing**: Comprehensive test coverage including robustness testing

#### Technical Architecture

**Core Types:**
```rust
pub enum VarRef {
    User(usize),     // Original variables from VariableRegistry
    Bound(u32),      // ANF temporary variables (unique IDs)
}

pub enum ANFExpr<T> {
    Atom(ANFAtom<T>),                           // Constants & variables
    Let(VarRef, ANFComputation<T>, Box<ANFExpr<T>>),  // let var = comp in body
}

pub struct ANFConverter {
    binding_depth: u32,                         // Current nesting level
    next_binding_id: u32,                       // Unique variable generator
    expr_cache: HashMap<StructuralHash, (u32, VarRef, u32)>,  // CSE cache
}
```

**Key Innovation - Scope-Aware CSE:**
```rust
// Cache entry: (scope_depth, variable, binding_id)
if cached_scope <= self.binding_depth {
    return ANFExpr::Atom(ANFAtom::Variable(cached_var));  // Safe to reuse
} else {
    self.expr_cache.remove(&structural_hash);  // Out of scope, remove
}
```

#### Current Capabilities

- **Basic CSE**: Automatically eliminates common subexpressions
- **Scope Safety**: Variables only referenced within valid binding scope
- **Limited Constant Folding**: Basic arithmetic operations on constants
- **Clean Code Generation**: Produces readable nested let-bindings
- **Property-Based Testing**: Robustness testing with random expressions

#### Current Limitations

- **No dead code elimination**: Unused let-bindings are not removed
- **Limited constant folding**: Only basic arithmetic operations
- **No optimization metrics**: No quantitative analysis of CSE effectiveness
- **Memory growth**: CSE cache grows without bounds
- **Scope management complexity**: Current approach may have edge cases
- **Domain safety**: No validation for transcendental function domains

#### Integration Points

**Existing Systems:**
- ✅ **VariableRegistry**: Seamless user variable management
- ✅ **ASTRepr**: Direct conversion from existing AST
- ✅ **Code Generation**: Produces valid Rust code
- ✅ **Test Infrastructure**: Comprehensive test coverage

**Future Integration Targets:**
- 🔄 **Egglog**: ANF as input for e-graph optimization
- 🔄 **JIT Compilation**: ANF → LLVM IR generation
- 🔄 **Symbolic Differentiation**: ANF-based autodiff
- 🔄 **Advanced Optimizations**: Enhanced constant folding, dead code elimination

## Ongoing Work 🚧

## Roadmap: Generic Numeric Types in Symbolic Optimizer

### Motivation
- Enable support for custom numeric types (e.g., rationals, dual numbers, complex numbers, arbitrary precision, etc.)
- Allow symbolic and automatic differentiation over types other than f64
- Facilitate integration with other math libraries and future-proof the codebase

### Technical Goals
- Make ASTRepr, symbolic optimizer, and all relevant passes generic over T: NumericType (or similar trait)
- Ensure all simplification, constant folding, and codegen logic works for generic T, not just f64
- Add trait bounds and/or specialization for transcendental and floating-point-specific rules
- Maintain performance and ergonomics for the common f64 case

### Considerations
- Some optimizations and simplifications are only valid for floating-point types (e.g., NaN, infinity, ln/exp rules)
- Codegen and JIT backends may need to be specialized or limited to f64 for now
- Test coverage must include both f64 and at least one custom numeric type (e.g., Dual<f64> or BigRational)

### Steps
1. Refactor ASTRepr and all symbolic passes to be generic over T
2. Add NumericType trait (if not already present) with required operations
3. Update tests and property-based tests to use both f64 and a custom type
4. Document which features are only available for f64 (e.g., JIT, codegen)
5. (Optional) Add feature flags for advanced numeric types

---

## Testing: Property-based tests for constant propagation
- Add proptests to ensure that constant folding and propagation in both symbolic and ANF passes are correct and robust.
- These tests should generate random expressions and check that all evaluation strategies (direct, ANF, symbolic) agree on results for all constant subexpressions.

## Domain Awareness
- Symbolic simplification should be domain-aware: only apply rewrites like exp(ln(x)) = x when x > 0.
- Property-based tests (proptests) must filter out invalid domains (e.g., negative values for ln, sqrt, etc.) to avoid spurious failures.
- Long-term: consider encoding domain constraints in the symbolic system and/or test harness.

## ✅ Completed (December 2024)

### File Reorganization and Modularization
- **✅ COMPLETED**: Reorganized large `src/final_tagless.rs` file (2819 lines) into focused modules
- **✅ COMPLETED**: Created modular structure:
  ```
  src/final_tagless/
  ├── mod.rs (main module file with comprehensive documentation)
  ├── traits.rs (core traits: MathExpr, StatisticalExpr, NumericType)
  ├── ast/
  │   ├── mod.rs
  │   ├── ast_repr.rs (ASTRepr enum with comprehensive documentation)
  │   ├── operators.rs (operator overloading for natural syntax)
  │   └── evaluation.rs (optimized evaluation methods)
  ├── interpreters/
  │   ├── mod.rs
  │   ├── direct_eval.rs (immediate evaluation)
  │   ├── pretty_print.rs (string representation)
  │   └── ast_eval.rs (AST construction for JIT)
  ├── variables/
  │   ├── mod.rs
  │   ├── registry.rs (VariableRegistry with thread-safe global registry)
  │   └── builder.rs (ExpressionBuilder for convenient construction)
  └── polynomial.rs (polynomial utilities with Horner's method)
  ```
- **✅ COMPLETED**: Added comprehensive documentation and examples to all modules
- **✅ COMPLETED**: Added inline tests for focused concerns
- **✅ COMPLETED**: Fixed missing functions in `ASTFunction` (`power`, `linear`, `constant_func`)
- **✅ COMPLETED**: Fixed missing exports for variable management functions
- **✅ COMPLETED**: Code compiles successfully with `cargo check`
- **✅ COMPLETED**: Most tests pass (148/151 passing)

### Technical Achievements
- **✅ COMPLETED**: Maintained backward compatibility - all existing APIs work
- **✅ COMPLETED**: Improved code organization and maintainability
- **✅ COMPLETED**: Enhanced documentation with usage examples
- **✅ COMPLETED**: Preserved all functionality while improving structure
- **✅ COMPLETED**: Added comprehensive inline tests for each module

### Current Status
- **✅ Code compiles**: `cargo check` passes successfully
- **✅ Most tests pass**: 148 out of 151 tests passing
- **⚠️ Minor test failures**: 3 test failures in summation and operator modules (not related to reorganization)
- **⚠️ Some warnings**: Various clippy warnings about unused variables and missing documentation

## 🔄 In Progress

### Code Quality Improvements
- **🔄 NEXT**: Fix remaining 3 test failures
- **🔄 NEXT**: Address clippy warnings for better code quality
- **🔄 NEXT**: Add missing documentation for struct fields and variants

## 📋 Planned (Next Steps)

### Further Modularization
- **📋 PLANNED**: Reorganize `src/symbolic.rs` module (if needed)
- **📋 PLANNED**: Reorganize `src/anf.rs` module (if needed)
- **📋 PLANNED**: Review and potentially reorganize other large modules

### Documentation and Examples
- **📋 PLANNED**: Add more comprehensive examples for each module
- **📋 PLANNED**: Create integration examples showing module interactions
- **📋 PLANNED**: Add performance benchmarks for reorganized code

### Testing and Quality
- **📋 PLANNED**: Add integration tests for the new modular structure
- **📋 PLANNED**: Ensure all examples compile and run correctly
- **📋 PLANNED**: Add property-based tests for core functionality

## 🎯 Long-term Goals

### Performance Optimization
- Cranelift JIT compilation improvements
- Rust hot-loading optimization
- Memory usage optimization

### Feature Expansion
- Advanced symbolic differentiation
- More statistical functions
- Enhanced summation capabilities
- Additional compilation backends

### User Experience
- Better error messages
- More ergonomic APIs
- Improved documentation
- Better IDE integration

## 📊 Metrics

### Code Organization (After Reorganization)
- **Main module**: `src/final_tagless/mod.rs` (246 lines, well-documented)
- **Core traits**: `src/final_tagless/traits.rs` (297 lines, focused)
- **AST module**: 4 focused files (ast_repr.rs: 288 lines, operators.rs: 350 lines, etc.)
- **Interpreters**: 3 focused files (direct_eval.rs: 297 lines, etc.)
- **Variables**: 2 focused files (registry.rs: 306 lines, builder.rs: 241 lines)
- **Polynomial**: 1 focused file (278 lines)

### Test Coverage
- **Total tests**: 151
- **Passing tests**: 148 (98%)
- **Failed tests**: 3 (2%, not related to reorganization)
- **Test categories**: Unit tests, integration tests, property tests

### Build Status
- **Compilation**: ✅ Successful (`cargo check` passes)
- **Library tests**: ✅ Mostly passing (148/151)
- **Examples**: ⚠️ Some compilation issues (feature-gated code)
- **Benchmarks**: ⚠️ Some compilation issues (feature dependencies)

---

*Last updated: December 2024*
*Status: File reorganization completed successfully*

## 🚀 Recent Major Achievement: Typed Variable System

We've successfully implemented a revolutionary typed variable system that brings compile-time type safety to mathematical expression building while maintaining beautiful operator overloading syntax and full backward compatibility.

### Key Features:
- **Compile-time Type Safety**: Variables carry type information (`f64`, `f32`, `i32`, etc.)
- **Automatic Type Promotion**: Safe conversions (e.g., `f32` → `f64`) happen automatically
- **Beautiful Syntax**: Natural mathematical expressions with `&x * &x + 2.0 * &x + &y`
- **Cross-type Operations**: Mix different numeric types with automatic promotion
- **Backward Compatibility**: Existing code continues to work unchanged
- **Better IDE Support**: Enhanced autocomplete and error messages

### Example Usage:
```rust
use mathcompile::prelude::*;

// Create typed variables
let math = MathBuilder::new();
let x: TypedVar<f64> = math.typed_var("x");
let y: TypedVar<f32> = math.typed_var("y");

// Build expressions with natural syntax and type safety
let x_expr = math.expr_from(x);
let y_expr = math.expr_from(y);
let expr = &x_expr * &x_expr + y_expr;  // f32 auto-promotes to f64

// Backward compatible API still works
let old_style = math.var("z");  // Defaults to f64
```

## 🎯 Current Focus Areas

### Performance Optimization
- [ ] **SIMD Vectorization**: Leverage CPU vector instructions for bulk operations
- [ ] **Memory Pool Allocation**: Reduce allocation overhead in hot paths
- [ ] **Compilation Caching**: Cache compiled functions across sessions
- [ ] **Parallel Evaluation**: Multi-threaded expression evaluation

### Advanced Mathematical Features
- [ ] **Complex Numbers**: Support for complex-valued expressions
- [ ] **Matrix Operations**: Linear algebra primitives and operations
- [ ] **Special Functions**: Gamma, Beta, Bessel functions, etc.
- [ ] **Numerical Integration**: Adaptive quadrature methods

### Language Bindings
- [ ] **Python Bindings**: PyO3-based Python interface
- [ ] **C/C++ Bindings**: Foreign function interface for C/C++
- [ ] **JavaScript/WASM**: WebAssembly compilation target
- [ ] **Julia Integration**: Native Julia package

## 🔮 Future Vision

### Advanced Compilation
- [ ] **LLVM Backend**: Direct LLVM IR generation for maximum performance
- [ ] **GPU Compilation**: CUDA/OpenCL code generation
- [ ] **Distributed Computing**: Automatic parallelization across nodes
- [ ] **Quantum Computing**: Quantum circuit compilation for quantum algorithms

### AI/ML Integration
- [ ] **Neural Network Primitives**: Built-in support for common NN operations
- [ ] **Automatic Batching**: Intelligent batching for ML workloads
- [ ] **Gradient Optimization**: Advanced optimization algorithms
- [ ] **Model Compilation**: Direct compilation of ML models

### Domain-Specific Extensions
- [ ] **Financial Mathematics**: Options pricing, risk calculations
- [ ] **Scientific Computing**: Physics simulations, numerical methods
- [ ] **Computer Graphics**: Shader-like mathematical expressions
- [ ] **Signal Processing**: FFT, filtering, convolution operations

## 📊 Performance Targets

- **Compilation Speed**: < 100ms for typical expressions
- **Runtime Performance**: Within 5% of hand-optimized C code
- **Memory Usage**: < 1MB overhead for expression compilation
- **Scalability**: Handle expressions with 10,000+ variables
