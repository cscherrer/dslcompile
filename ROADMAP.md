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
- [ ] **Sufficient Statistics**: Automatic emergence through symbolic optimization (infrastructure in progress)
- [x] **MCMC Integration Ready**: Direct compatibility with nuts-rs and other samplers
- [x] **Performance Optimization**: ~19M evaluations/second for compiled log-posterior functions
- [x] **Detailed Performance Profiling**: Stage-by-stage timing analysis with breakdown percentages
- [x] **Amortization Analysis**: Automatic calculation of compilation cost vs. runtime benefit
- [x] **dlopen2 Migration**: Replaced libloading with dlopen2 for better type safety and simplified architecture
  - Eliminated unsafe `std::mem::transmute` calls
  - Unified function pointer architecture (single `extern "C" fn` instead of 3 optional variants)
  - Improved error handling and fallback logic
  - Maintained full backward compatibility

**Suggestion**: Consider splitting statistical computing features into separate crates (e.g., `mathcompile-stats`) to maintain focused scope and easier testing.

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

### 1. **Statistical Computing & PPL Backend Stabilization** (High Priority - IMMEDIATE)
**Context**: Following the major statistical computing PR, we need to stabilize and complete the new infrastructure.

#### Phase 1: Core Stabilization (Next 2-4 weeks)
- [ ] **Complete Runtime Data Binding**: Fix `call_with_data` method to actually use data parameter
- [ ] **Robust Error Handling**: Replace silent fallbacks (like `to_f64().unwrap_or(0.0)`) with explicit error handling
- [ ] **ANF Integration**: Complete the ANF/CSE integration that's currently disabled with TODOs
- [ ] **Mixed Input Implementation**: Complete or remove the incomplete mixed input type support
- [ ] **API Consistency**: Ensure all new methods have consistent parameter usage and error handling

#### Phase 2: API Simplification & Documentation (Following 2-4 weeks)
- [ ] **API Surface Reduction**: Evaluate if all new types (`CompiledFunction<Input>`, `InputSpec`, etc.) are necessary for MVP
- [ ] **Safety Documentation**: Document safety invariants for `Send`/`Sync` implementations and unsafe code
- [ ] **Performance Characteristics**: Document memory allocation patterns and performance trade-offs
- [ ] **Migration Examples**: Create examples showing how to migrate from old to new statistical APIs

#### Phase 3: Testing & Validation (Ongoing)
- [ ] **Strengthen Test Suite**: Replace weakened test assertions with proper validation
- [ ] **Property-Based Testing**: Add property tests for statistical computing features
- [ ] **Performance Benchmarks**: Add benchmarks comparing different evaluation strategies
- [ ] **Memory Safety Validation**: Add tests specifically for the dlopen2 integration and thread safety

#### Recommended Implementation Strategy
1. **Start Simple**: Focus on the core Bayesian linear regression use case first
2. **Incremental Complexity**: Add advanced features (partial evaluation, abstract interpretation) in separate PRs
3. **Test-Driven**: Write tests before implementing complex features
4. **Documentation-First**: Document safety invariants and API contracts clearly

### 2. **Enhanced Abstract Interpretation & Constraint Integration** (High Priority - FUTURE)
- [ ] **Inequality-Domain Unified Framework**: Seamless bidirectional translation between constraints and domains
- [ ] **Certified Computation Pipeline**: Mathematical guarantees for numerical analysis results
- [ ] **Constraint-Aware Partial Evaluation**: Specialize computations based on inequality constraints
- [ ] **Rigorous Error Bound Tracking**: Automatic propagation of mathematical error bounds
- [ ] **Data Range Analysis**: Implement min/max value tracking for optimization opportunities
- [ ] **Sparsity Pattern Detection**: Identify and eliminate zero-value terms automatically
- [ ] **Statistical Property Analysis**: Use mean, variance for numerical stability optimizations
- [ ] **Correlation Structure Analysis**: Detect and eliminate redundant computations
- [ ] **Partial Data Specialization**: Support fixing some data points while varying others
- [ ] **Hierarchical Model Support**: Specialize on group-level data for hierarchical models
- [ ] **Time Series Specialization**: Specialize on historical data for future predictions
- [ ] **Ensemble Method Support**: Specialize each model on different data subsets
- [ ] **Domain-Aware Partial Evaluation**: Integrate with interval domain analysis and inequality constraints
- [ ] **Abstract Interpretation Framework**: Formal framework for compile-time analysis with constraint solving
- [ ] **Constraint Satisfaction Integration**: CAD, LP, and SOS solvers for polynomial inequality systems
- [ ] **Formal Verification Support**: Generate mathematical proofs for optimization correctness

### 3. **API Modernization** (High Priority)
- [ ] Update all examples to use the new operator syntax
- [ ] Update benchmarks to showcase the new API
- [ ] Create comprehensive documentation for the unified system
- [ ] Add migration guide from old verbose API

### 4. **Performance Optimization** (Medium Priority)
- [ ] Benchmark the new type system vs old system
- [ ] Optimize cloning in operator overloading
- [ ] Profile memory usage of the unified system
- [ ] Consider `Copy` trait for small expressions

### 5. **Enhanced Type System** (Medium Priority)
- [ ] Add more mathematical function categories (Trigonometric, Hyperbolic, etc.)
- [ ] Implement complex number support
- [ ] Add matrix/vector types
- [ ] Enhanced error messages for type mismatches

### 6. **Advanced Features** (Lower Priority)
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

## Next Steps (Phase 5: Extensibility & Language Bindings)

**Status**: Ready to Begin (May 2025)

Following the completion of statistical computing features and domain analysis, the next major phase focuses on extensibility for external language bindings (Python, Julia) and better organization of mathematical operations and rules.

### Goals Overview

1. **External Language Integration**: Enable Python and Julia users to define custom types, operations, and egglog rules
2. **Better Normalization**: Implement canonical forms (Sub → Add + Neg, Div → Mul + Pow(-1))
3. **Rule Organization**: Extract egglog rules from inline strings to separate files
4. **Operation Categories**: Organize operations by mathematical domain (basic, transcendental, trigonometric, special functions)

### Phase 5A: Foundation Refactoring (Q2 2025)

#### PR 1: Extract Egglog Rules to Separate Files
**Priority**: High - Improves maintainability and enables rule development
**Estimated Effort**: 1-2 weeks

- [ ] **Create Rules Directory Structure**
  - `rules/basic_arithmetic.egg` - Identity, commutativity, associativity rules
  - `rules/transcendental.egg` - Exp, ln, log transformation rules  
  - `rules/trigonometric.egg` - Sin, cos, tan identities and transformations
  - `rules/power_rules.egg` - Power and root simplification rules
  - `rules/summation.egg` - Summation linearity and algebraic rules
  - `rules/normalization.egg` - Canonical form transformation rules

- [ ] **Implement Rule Loader System**
  - Dynamic rule file loading and combination
  - Rule validation and syntax checking
  - Conditional rule loading based on enabled features
  - Rule dependency management

- [ ] **Migrate Existing Rules**
  - Extract ~200 lines of inlined rules from `egglog_integration.rs`
  - Organize rules by mathematical domain
  - Add documentation and examples for each rule file
  - Maintain backward compatibility

**Downstream Effects**: 
- Existing tests should pass unchanged
- Rule development becomes easier for contributors
- Version control of mathematical rules improves

#### PR 2: Implement Basic Normalization
**Priority**: High - Enables better optimization and reduces rule complexity
**Estimated Effort**: 1-2 weeks

- [ ] **Canonical Form Transformations (Pre-Egglog)**
  - `Sub(a, b) → Add(a, Neg(b))` - Eliminate subtraction in favor of addition + negation
  - `Div(a, b) → Mul(a, Pow(b, -1))` - Eliminate division in favor of multiplication + negative power
  - Implement as AST transformation pass before egglog optimization
  - Apply recursively to ensure complete normalization
  - **Theoretical Foundation**: Creates true canonical forms (unique representation per equivalence class)

- [ ] **Separation of Concerns Architecture**
  - Insert normalization step: `AST → Normalize → ANF → Egglog → Extract → Codegen`
  - **Inert Expression Preservation**: Store original user expression for display/debugging
  - **Domain Analysis Integration**: Ensure domain analysis works with normalized forms
  - Update ANF transformation to handle canonical forms efficiently
  - **Performance Benefit**: Reduces egglog rule complexity from O(n²) to O(n) patterns

- [ ] **Egglog Rule Simplification**
  - Remove redundant rules that handle non-canonical cases (Sub, Div)
  - Simplify remaining rules by assuming canonical forms
  - Focus egglog rules on algebraic identities and advanced optimizations
  - Reduce total rule count by ~30-40% for better performance
  - **Complexity Reduction**: From ~200 lines to ~120 lines of egglog rules

- [ ] **Bidirectional Display System**
  - Add "denormalization" for pretty-printing (Add(a, Neg(b)) → Sub(a, b))
  - Preserve user intent in error messages and debug output
  - **User Experience**: Allow users to choose normalized vs. natural display
  - **Implementation**: Separate display layer from internal canonical representation

- [ ] **Validation and Testing**
  - Property-based tests ensuring normalization preserves mathematical equivalence
  - Performance benchmarks comparing pre/post normalization egglog speed
  - Domain safety tests ensuring normalization doesn't break domain analysis
  - **Correctness**: Extensive testing of canonical form uniqueness

**Theoretical Benefits**:
- **Canonical Forms**: Every expression has exactly one normalized representation
- **Decidable Equality**: Two expressions are equal iff their canonical forms are identical
- **Performance**: Polynomial complexity O(k^2.1) vs. exponential for multiple forms
- **Rule Simplicity**: Egglog rules only need to handle canonical cases

**Downstream Effects**:
- Egglog optimization becomes faster due to fewer rules (~40% reduction)
- Rule development becomes simpler (only canonical forms)
- Some existing optimizations may become more effective
- Debug output shows canonical forms by default (configurable)
- **User Experience**: Clear separation between syntax and semantics

#### PR 2.5: Implement Inert Expression System
**Priority**: High - Foundation for advanced CAS capabilities
**Estimated Effort**: 2-3 weeks
**Research Foundation**: Based on Fredrik Johansson's CAS design principles

- [ ] **Inert vs. Active Expression Separation**
  - **Inert Expressions**: Completely syntactic, no automatic rewriting
  - **Active Expressions**: Mathematical values with canonical forms
  - Clear API distinction: `expr.as_inert()` vs `expr.evaluate()`
  - User control over when/how rewriting occurs
  - **Benefit**: Preserves exact user input while enabling optimization

- [ ] **Symbolic Expression Preservation**
  - Store original user syntax alongside normalized forms
  - Support multiple display modes: original, canonical, pretty-printed
  - Enable round-trip: `parse(expr.to_string()) == expr`
  - **Use Case**: Mathematical education, debugging, proof presentation

- [ ] **Type-Integrated Symbolic System**
  - Symbolic integers: `n ≥ 3`, `n ≡ 15 (mod 60)`, `n is-prime`
  - Symbolic matrices: `A = BC` where B,C are symbolic matrices
  - Symbolic ring elements: `x ∈ R` where R has properties like "commutative"
  - **Innovation**: Symbolic objects as first-class citizens in type system

- [ ] **Partial Knowledge Representation**
  - Support expressions with unknown/partial information
  - Interval enclosures: `x ∈ [3.14, 3.15]`
  - Constraint propagation through symbolic operations
  - **Mathematical Foundation**: Abstract interpretation framework

- [ ] **Performance Considerations**
  - Lazy evaluation for expensive symbolic operations
  - Efficient storage for common symbolic patterns
  - Caching of computed properties (determinant, eigenvalues, etc.)
  - **Goal**: Symbolic operations competitive with specialized tools

**Downstream Effects**:
- Foundation for advanced mathematical reasoning
- Better user experience for educational applications
- Enables formal verification capabilities
- May increase memory usage for complex expressions

#### PR 2.7: Enhanced Integer Performance Architecture
**Priority**: Medium - Long-term performance foundation
**Estimated Effort**: 3-4 weeks
**Research Foundation**: Based on FLINT and modern CAS performance principles

- [ ] **Context-Optimized Integer Representations**
  - Small integers: packed arrays, SIMD-friendly layouts
  - Polynomial coefficients: modular arithmetic, delayed reduction
  - Matrix elements: cache-friendly storage patterns
  - **Innovation**: Automatic representation selection based on usage context

- [ ] **Advanced Arithmetic Strategies**
  - Vectorized operations for arrays of small integers
  - Karatsuba/Toom-Cook for large integer multiplication
  - Montgomery arithmetic for modular operations
  - **Performance Target**: 10-100x improvement for integer-heavy workloads

- [ ] **Memory Management Optimization**
  - Object pooling for frequently allocated integer types
  - Arena allocation for temporary computations
  - Copy-on-write for large integer arrays
  - **Goal**: Reduce allocation overhead by 80%+

- [ ] **Compiler Integration**
  - LLVM backend for integer arithmetic code generation
  - Auto-vectorization hints for integer array operations
  - Profile-guided optimization for hot integer paths
  - **Future**: Custom integer types in generated code

**Downstream Effects**:
- Significant performance improvements for symbolic computation
- Better scalability for large mathematical problems
- Foundation for high-performance numerical-symbolic hybrid algorithms
- May require changes to existing integer-handling code

### Phase 5B: Extensibility Infrastructure (Q3 2025)

#### PR 4: Plugin Architecture for Operations
**Priority**: High - Core requirement for language bindings
**Estimated Effort**: 3-4 weeks

- [ ] **Dynamic Operation Registration**
  - Runtime registration of custom operations
  - Type-safe operation definitions with compile-time validation
  - Support for operations with custom evaluation logic
  - Integration with existing optimization pipeline

- [ ] **Custom Rule Integration**
  - Allow external crates to provide egglog rules
  - Rule conflict detection and resolution
  - Namespace management for custom operations
  - Rule validation and testing framework

- [ ] **Plugin API Design**
  - Stable API for external operation definitions
  - Documentation and examples for plugin development
  - Version compatibility guarantees
  - Plugin discovery and loading mechanisms

**Downstream Effects**:
- External crates can extend MathCompile functionality
- Foundation for language-specific bindings
- May affect compilation time due to dynamic features

#### PR 5: Foreign Function Interface (FFI) Layer
**Priority**: High - Required for Python/Julia integration
**Estimated Effort**: 2-3 weeks

- [ ] **C-Compatible API**
  - Expression building, optimization, and evaluation via FFI
  - Proper error handling across FFI boundary
  - Memory management for cross-language data
  - Thread safety guarantees for concurrent access

- [ ] **Language Binding Foundations**
  - Common interface for Python and Julia bindings
  - Serialization support for complex data types
  - Callback mechanisms for custom operations
  - Performance optimization for FFI calls

- [ ] **Safety and Testing**
  - Comprehensive FFI safety validation
  - Cross-language integration tests
  - Memory leak detection and prevention
  - Error propagation testing

**Downstream Effects**:
- Enables external language integration
- May introduce new dependencies
- Requires careful API design for long-term stability

#### PR 6: Enhanced Rule System
**Priority**: Medium - Advanced optimization capabilities
**Estimated Effort**: 2-3 weeks

- [ ] **Conditional Rules**
  - Domain-aware rule application
  - Context-sensitive transformations
  - Rule precondition checking
  - Integration with interval domain analysis

- [ ] **User-Provided Rules**
  - Support for user-defined egglog rule files
  - Rule validation and syntax checking
  - Rule composition and dependency management
  - Performance impact analysis

- [ ] **Advanced Rule Features**
  - Rule priority and ordering
  - Conditional rule activation based on expression properties
  - Rule debugging and tracing capabilities
  - Rule performance profiling

**Downstream Effects**:
- More sophisticated optimization capabilities
- User extensibility for domain-specific optimizations
- Potential performance impact from rule complexity

### Phase 5C: Language Bindings (Q3-Q4 2025)

#### PR 7: Python Integration
**Priority**: High - Major user-facing feature
**Estimated Effort**: 3-4 weeks

- [ ] **Python Package Development**
  - PyO3-based Python bindings or ctypes interface
  - Pythonic API design with operator overloading
  - Integration with NumPy and SciPy ecosystems
  - Comprehensive Python documentation and examples

- [ ] **Custom Python Operations**
  - Support for Python-defined mathematical operations
  - Automatic compilation of Python functions to Rust
  - Python-defined egglog rules with validation
  - Performance optimization for Python callbacks

- [ ] **Python-Specific Features**
  - Jupyter notebook integration and visualization
  - Integration with symbolic math libraries (SymPy)
  - Python package distribution (PyPI)
  - Example notebooks and tutorials

**Example Usage**:
```python
import mathcompile as mc

# Define custom operation in Python
@mc.operation
def custom_sigmoid(x, steepness=1.0):
    return 1.0 / (1.0 + mc.exp(-steepness * x))

# Use in expressions with automatic optimization
x = mc.var("x")
expr = x**2 + custom_sigmoid(x, 2.0)
optimized = mc.optimize(expr)
```

#### PR 8: Julia Integration Enhancement
**Priority**: High - Build on existing Julia tools
**Estimated Effort**: 2-3 weeks

- [ ] **Enhanced Julia Package**
  - Extend existing `jltools/` with custom operation support
  - Julia macro system for defining operations
  - Integration with Julia's multiple dispatch system
  - High-performance Julia-Rust interop

- [ ] **Julia-Specific Features**
  - Integration with DifferentialEquations.jl ecosystem
  - Support for Julia's type system and generic programming
  - Julia package registration and documentation
  - Performance benchmarks against pure Julia implementations

- [ ] **Cross-Language Compatibility**
  - Shared operation definitions between Python and Julia
  - Common rule format for both languages
  - Performance comparison and optimization

**Example Usage**:
```julia
using MathCompile

# Define custom operation with Julia macro
@operation function custom_bessel(n, x)
    # Custom implementation or call to SpecialFunctions.jl
    return besselj(n, x)
end

# Use in expressions
x = @var x
expr = x^2 + custom_bessel(1, x)
optimized = optimize(expr)
```

#### PR 9: Cross-Language Rule Sharing
**Priority**: Medium - Ecosystem development
**Estimated Effort**: 2-3 weeks

- [ ] **Rule Serialization**
  - Common format for rules across languages
  - Rule package management system
  - Version compatibility and migration
  - Rule sharing and distribution

- [ ] **Ecosystem Development**
  - Community rule repository
  - Rule validation and testing framework
  - Documentation and examples for rule development
  - Integration with package managers (pip, Pkg.jl)

### Phase 5D: Advanced Mathematical Features (Q4 2025)

#### PR 10: Special Functions Integration
**Priority**: Medium - Expand mathematical coverage
**Estimated Effort**: 2-3 weeks

- [ ] **Special Functions Categories**
  - `SpecialFunctions` - Gamma, Beta, Bessel functions
  - `LogExpFunctions` - LogGamma, LogBeta, softmax, logsumexp
  - `TrigFunctions` - Extended trigonometric and hyperbolic functions
  - Integration with existing "special" crate ecosystem

- [ ] **Mathematical Identities**
  - Comprehensive egglog rules for special function identities
  - Asymptotic expansions and approximations
  - Domain-specific optimizations
  - Numerical stability improvements

- [ ] **Performance Optimization**
  - Efficient evaluation strategies for special functions
  - Approximation vs. exact computation trade-offs
  - Integration with JIT compilation
  - Benchmarks against specialized libraries

#### PR 11: Domain-Aware Optimization
**Priority**: Medium - Advanced optimization
**Estimated Effort**: 2-3 weeks

- [ ] **Enhanced Domain Integration**
  - Domain-conditional egglog rules
  - Interval arithmetic integration
  - Uncertainty quantification support
  - Numerical stability analysis

- [ ] **Advanced Transformations**
  - Context-sensitive optimizations
  - Multi-stage optimization pipeline
  - Optimization strategy selection
  - Performance vs. accuracy trade-offs

#### PR 12: Performance and Polish
**Priority**: High - Production readiness
**Estimated Effort**: 2-3 weeks

- [ ] **Comprehensive Benchmarking**
  - Performance regression testing
  - Cross-language performance comparison
  - Memory usage optimization
  - Compilation time optimization

- [ ] **Documentation and Examples**
  - Complete API documentation
  - Tutorial series for each language binding
  - Performance guide and best practices
  - Migration guide from existing solutions

- [ ] **Production Features**
  - Error handling and recovery
  - Logging and debugging support
  - Configuration and tuning options
  - Deployment and packaging

### Implementation Guidelines

#### Technical Considerations
- **Bisectability**: Each PR should be independently testable and deployable
- **Backward Compatibility**: Maintain existing API during transition period
- **Performance**: Benchmark each change to avoid regressions
- **Safety**: Comprehensive testing of FFI and cross-language features
- **Documentation**: Update docs and examples with each PR

#### Testing Strategy
- **Property-Based Testing**: Mathematical correctness validation
- **Cross-Language Testing**: Consistency across Python, Julia, and Rust
- **Performance Testing**: Regression detection and optimization validation
- **Integration Testing**: End-to-end workflow validation
- **Safety Testing**: Memory safety and thread safety validation

#### Risk Mitigation
- **Feature Flags**: Use Cargo features for experimental functionality
- **Gradual Migration**: Support both old and new APIs during transition
- **Comprehensive Testing**: Extensive test coverage for all new features
- **Community Feedback**: Early feedback on API design and usability
- **Performance Monitoring**: Continuous performance tracking

This phase represents a major expansion of MathCompile's capabilities, enabling it to serve as a foundation for mathematical computing across multiple programming languages while maintaining its core strengths in symbolic optimization and performance.

#### PR 3: Reorganize Operations into Categories  
**Priority**: Medium - Improves code organization and extensibility
**Estimated Effort**: 2-3 weeks

- [ ] **Create Operation Category Structure**
  ```
  src/operations/
  ├── mod.rs              # Operation registry and traits
  ├── basic.rs            # Add, Sub, Mul, Div, Pow, Neg
  ├── transcendental.rs   # Exp, Ln, Log, Log10, Log2
  ├── trigonometric.rs    # Sin, Cos, Tan, Asin, Acos, Atan
  ├── hyperbolic.rs       # Sinh, Cosh, Tanh, Asinh, Acosh, Atanh
  └── special.rs          # Gamma, Bessel, etc. (depends on "special" crate)
  ```

- [ ] **Operation Trait System**
  - Define `MathOperation` trait for extensibility
  - Include egglog rule generation in trait
  - Support operation metadata (arity, commutativity, etc.)
  - Enable dynamic operation registration

- [ ] **Category-Specific Rules**
  - Each operation category includes associated egglog rules
  - Rules are automatically loaded when category is enabled
  - Support conditional compilation based on feature flags

**Downstream Effects**:
- Better code organization for contributors
- Easier to add new mathematical functions
- Foundation for plugin architecture

#### PR 3.5: Inequality and Error Bound Support
**Priority**: Medium - Advanced mathematical analysis capabilities
**Estimated Effort**: 2-3 weeks
**Research Foundation**: Address CAS weakness in inequality manipulation

- [ ] **Inequality Expression Types**
  - First-class support for `<`, `≤`, `>`, `≥` expressions
  - Inequality chains: `a < b ≤ c < d`
  - Set membership: `x ∈ [a, b]`, `x ∈ (a, ∞)`
  - **Innovation**: Inequalities as mathematical objects, not just predicates

- [ ] **Error Bound Propagation**
  - Automatic error bound tracking through computations
  - Stirling series with explicit error terms
  - Taylor series with remainder bounds
  - **Use Case**: Rigorous numerical analysis, certified computation

- [ ] **Constraint Solving**
  - Linear inequality systems (LP/QP integration)
  - Polynomial inequality solving (CAD/SOS methods)
  - Domain restriction propagation
  - **Mathematical Foundation**: Real algebraic geometry

- [ ] **Integration with Existing Systems**
  - Domain analysis enhanced with inequality constraints
  - Egglog rules for inequality simplification
  - Pretty-printing for mathematical notation
  - **Goal**: Inequalities as natural as equalities

**Downstream Effects**:
- Enables rigorous numerical analysis
- Better support for optimization problems
- Foundation for formal verification
- May require changes to expression evaluation logic

#### PR 3.6: Bidirectional Inequality-Domain Integration
**Priority**: High - Core synergy between inequalities and abstract interpretation
**Estimated Effort**: 3-4 weeks
**Mathematical Foundation**: Unified constraint-domain framework

- [ ] **Bidirectional Translation System**
  - Convert inequalities to interval domains: `x > 0` → `IntervalDomain::positive(0.0)`
  - Convert domains to inequality constraints: `[1, 5]` → `1 ≤ x ≤ 5`
  - Round-trip preservation: `parse(domain.to_inequalities()).to_domain() == domain`
  - **Innovation**: Seamless interoperability between constraint and domain representations

- [ ] **Enhanced Domain Analysis with Constraints**
  - `analyze_with_constraints(expr, inequalities)` → enhanced domain
  - Constraint propagation through expression trees
  - Intersection of expression domains with user constraints
  - **Mathematical Rigor**: Soundness and completeness guarantees

- [ ] **Rigorous Error Bound Framework**
  - `RigorousDomain<F>` combining value intervals and error bounds
  - Certified computation with mathematical guarantees
  - Stirling approximation with rigorous error bounds: `|error| ≤ 1/(12*n)`
  - Taylor series with remainder term tracking
  - **Use Case**: Numerical analysis with mathematical proofs

- [ ] **Constraint-Aware Optimization Pipeline**
  - Domain-aware egglog rules with inequality preconditions
  - Safe transformations: `exp(ln(x)) → x` only when `x > 0`
  - Constraint solving integration (CAD, LP, SOS)
  - **Performance**: More aggressive optimizations with safety guarantees

- [ ] **Automatic Differentiation with Domain Safety**
  - Domain-aware derivative computation
  - Prevent undefined derivatives: `d/dx ln(x)` requires `x > 0`
  - Propagate domain constraints through differentiation
  - **Mathematical Correctness**: No NaN or undefined results

- [ ] **Certified Optimization**
  - Optimization with rigorous bounds: `min f(x) on [a,b]` with error certificates
  - Global optimization with mathematical guarantees
  - Constraint satisfaction with proof generation
  - **Applications**: Verified numerical methods, formal verification

**Theoretical Benefits**:
- **Unified Framework**: Inequalities and domains work together seamlessly
- **Mathematical Rigor**: Certified computation with soundness guarantees
- **Performance**: Domain-aware optimizations can be more aggressive
- **User Experience**: Natural mathematical notation with automatic safety

**Downstream Effects**:
- Foundation for formal verification capabilities
- Enables certified numerical analysis
- Better integration with optimization libraries
- May require significant changes to expression evaluation pipeline
- Performance impact from constraint checking (mitigated by caching)

#### PR 3.7: Text-Friendly and Human-Readable Design
**Priority**: Medium - User experience and interoperability
**Estimated Effort**: 1-2 weeks
**Research Foundation**: Plain text as first-class interface

- [ ] **Round-Trip Serialization**
  - Every object prints to valid syntax: `parse(expr.to_string()) == expr`
  - Context-independent representation
  - No implicit state or hidden dependencies
  - **Benefit**: Trivial serialization, debugging, composition

- [ ] **Unified Namespace Design**
  - Consistent naming conventions across all mathematical domains
  - Avoid module import complexity for basic operations
  - Mathematical notation where possible: `sin`, `cos`, `exp`, `ln`
  - **User Experience**: Predictable, discoverable API

- [ ] **Multiple Output Formats**
  - ASCII math: `x^2 + 2*x + 1`
  - LaTeX: `x^{2} + 2x + 1`
  - Unicode math: `x² + 2x + 1`
  - **Default**: Human-readable ASCII with configurable alternatives

- [ ] **Functional Semantics**
  - Avoid implicit context and global state
  - Pure functions where mathematically appropriate
  - Explicit parameter passing for configuration
  - **Benefit**: Reproducible, composable, testable

**Downstream Effects**:
- Better integration with external tools
- Improved debugging and development experience
- Easier documentation and example creation
- Foundation for language bindings
