# MathCompile Development Roadmap

## Project Overview

MathCompile is a mathematical expression compiler that transforms symbolic mathematical expressions into optimized, executable code. The project aims to bridge the gap between mathematical notation and high-performance computation.

## Core Architecture Insight (May 31, 2025)

**Key Simplification**: Statistical functionality should be a special case of the general mathematical expression system, not a separate specialized system.

- ✅ **General system works**: `call_multi_vars()` handles all cases correctly
- ✅ **Statistical computing**: Works by flattening parameters and data into a single variable array
- ✅ **Example created**: `simplified_statistical_demo.rs` demonstrates the approach
- ✅ **Core methods fixed**: `call_with_data()` now properly concatenates params and data
- ✅ **Architecture validated**: Statistical functions are just mathematical expressions with more variables
- ✅ **Legacy code removed**: Cleaned up unused statistical specialization types and methods

**Status**: Core simplification **COMPLETED**. System is now unified and clean.

## 🚀 Next Steps (Ordered by Core → Downstream)

### 1. Basic Normalization (Foundation for Everything Else) ✅ COMPLETED

#### Canonical Form Transformations ✅ COMPLETED
- [x] **Canonical Form Transformations**: Complete implementation of `Sub(a, b) → Add(a, Neg(b))` and `Div(a, b) → Mul(a, Pow(b, -1))`
- [x] **Pipeline Integration**: Full normalization pipeline: `AST → Normalize → ANF → Egglog → Extract → Codegen`
- [x] **Bidirectional Processing**: Normalization for optimization and denormalization for display
- [x] **Egglog Rule Simplification**: Achieved ~40% rule complexity reduction with canonical-only rules
- [x] **Comprehensive Testing**: 12 test functions covering all normalization aspects
- [x] **Test Infrastructure**: Fixed hanging test issues and ensured robust test execution

**Status**: ✅ **COMPLETED** (May 31, 2025)

**Implementation Details**:
- Created `src/ast/normalization.rs` with comprehensive normalization functions
- Implemented `normalize()`, `denormalize()`, `is_canonical()`, and `count_operations()` functions
- Updated egglog integration to use canonical-only rules in `rules/canonical_arithmetic.egg`
- Modified `src/symbolic/egglog_integration.rs` to implement full pipeline: AST → Normalize → Egglog → Extract → Denormalize
- Created comprehensive test suite with 12 test functions covering all aspects of normalization
- Demonstrated ~40% rule complexity reduction by eliminating Sub/Div handling from egglog rules
- Fixed hanging test issue in `test_egglog_integration_with_normalization`

**Benefits Achieved**:
- Simplified egglog rules: Only need to handle Add, Mul, Neg, Pow (not Sub, Div)
- Consistent patterns: All operations follow additive/multiplicative patterns
- Better optimization opportunities: More algebraic simplification possibilities
- Reduced complexity: ~40% fewer rule cases to maintain and debug
- Foundation established for all subsequent optimization improvements

### 2. Rule System Organization ✅ COMPLETED

#### Extract Egglog Rules to Files ✅ COMPLETED
- [x] **Create Rules Directory**: Separate files for `basic_arithmetic.egg`, `transcendental.egg`, `trigonometric.egg`, etc.
- [x] **Rule Loader System**: Dynamic rule file loading, validation, and combination
- [x] **Migrate Existing Rules**: Extract ~200 lines of inlined rules from code to organized files
- [x] **Rule Documentation**: Add examples and documentation for each rule category

#### Enhanced Rule System ✅ COMPLETED
- [x] **EgglogOptimizer Integration**: Complete integration with RuleLoader for dynamic rule loading
- [x] **Multiple Configurations**: Support for Default, Domain-Aware, and Canonical-Only optimizers
- [x] **Rule Information API**: `rule_info()` method to inspect loaded rule categories
- [x] **Custom Rule Configurations**: Support for user-defined rule category combinations

**Status**: ✅ **COMPLETED** (May 31, 2025)

**Implementation Details**:
- Created organized rule files in `rules/` directory: `core_datatypes.egg`, `basic_arithmetic.egg`, `domain_aware_arithmetic.egg`, `transcendental.egg`, `trigonometric.egg`, `summation.egg`
- Implemented `RuleLoader` with `RuleConfig` for flexible rule management
- Integrated `RuleLoader` with `EgglogOptimizer` replacing inline rules with dynamic loading
- Added multiple optimizer constructors: `new()`, `with_rule_config()`, `domain_aware()`, `canonical_only()`
- Created comprehensive example `rule_loader_demo.rs` demonstrating all features
- Resolved rule conflicts by removing duplicates between files
- Added `rule_info()` API for inspecting loaded rule categories

**Benefits Achieved**:
- Modular rule organization by mathematical domain
- Dynamic rule loading with validation and error handling
- Flexible optimizer configurations for different use cases
- Clean separation between rule definitions and optimizer logic
- Foundation for user-provided rules and plugin architecture
- Comprehensive testing and documentation

### 3. Native egglog Integration ✅
**Status**: COMPLETED (2025-05-31)
- ✅ Implemented `NativeEgglogOptimizer` using egglog directly
- ✅ Created comprehensive mathematical rule set
- ✅ Added AST to egglog s-expression conversion
- ✅ Implemented canonical form support (Sub → Add + Neg, Div → Mul + Pow(-1))
- ✅ Fixed f64 formatting for egglog compatibility
- ✅ Added comprehensive test suite (9/9 tests passing)
- ✅ Created domain-aware optimization demo
- ✅ Established foundation for advanced domain analysis

**Key Achievement**: Discovered that egglog itself provides native abstract interpretation capabilities, making manual integration unnecessary and opening up powerful optimization possibilities following the Herbie paper approach.

## Current Priority 🎯

### 4. Advanced Domain-Aware Optimization
**Status**: FOUNDATION COMPLETE - Ready for Full Implementation
**Target**: Q2 2025

### 5. ANF Integration Completion

#### Complete ANF-Domain Integration
- [ ] **Domain-Aware ANF**: Integrate domain analysis into A-Normal Form transformations
- [ ] **Safe CSE**: Ensure common subexpression elimination respects domain constraints
- [ ] **ANF Integration**: Complete the ANF/CSE integration that's currently disabled with TODOs
- [ ] **Optimization Metrics**: Track domain safety improvements in ANF pipeline

### 6. Advanced Domain Analysis

#### Inequality and Constraint Integration
- [ ] **Inequality Expression Types**: First-class support for `<`, `≤`, `>`, `≥` expressions and set membership
- [ ] **Bidirectional Translation**: Convert inequalities ↔ interval domains seamlessly
- [ ] **Constraint-Aware Optimization**: Domain-aware egglog rules with inequality preconditions
- [ ] **Error Bound Propagation**: Automatic error bound tracking through computations

#### Enhanced Abstract Interpretation
- [ ] **Certified Computation Pipeline**: Mathematical guarantees for numerical analysis results
- [ ] **Constraint-Aware Partial Evaluation**: Specialize computations based on inequality constraints
- [ ] **Rigorous Error Bound Tracking**: Automatic propagation of mathematical error bounds

### 7. Operation System Reorganization

#### Reorganize Operations into Categories
- [ ] **Operation Category Structure**: `src/operations/` with `basic.rs`, `transcendental.rs`, `trigonometric.rs`, etc.
- [ ] **Operation Trait System**: Define `MathOperation` trait for extensibility with egglog rule generation
- [ ] **Category-Specific Rules**: Each operation category includes associated egglog rules
- [ ] **Dynamic Registration**: Enable runtime registration of custom operations

#### Special Functions Integration
- [ ] **Special Functions Categories**: Gamma, Beta, Bessel functions with mathematical identities
- [ ] **Performance Optimization**: Efficient evaluation strategies and approximation trade-offs
- [ ] **Integration**: Work with existing "special" crate ecosystem

### 8. Extensibility Infrastructure

#### Plugin Architecture
- [ ] **Dynamic Operation Registration**: Runtime registration with type-safe operation definitions
- [ ] **Custom Rule Integration**: Allow external crates to provide egglog rules with conflict detection
- [ ] **Plugin API Design**: Stable API with documentation and version compatibility guarantees

#### Foreign Function Interface (FFI)
- [ ] **C-Compatible API**: Expression building, optimization, and evaluation via FFI
- [ ] **Language Binding Foundations**: Common interface for Python and Julia bindings
- [ ] **Safety and Testing**: Comprehensive FFI safety validation and cross-language integration tests

### 9. Language Bindings

#### Python Integration
- [ ] **Python Package**: PyO3-based bindings with Pythonic API and NumPy integration
- [ ] **Custom Python Operations**: Support for Python-defined mathematical operations and egglog rules
- [ ] **Python-Specific Features**: Jupyter integration, SymPy compatibility, PyPI distribution

#### Julia Integration Enhancement
- [ ] **Enhanced Julia Package**: Extend existing `jltools/` with custom operation support
- [ ] **Julia-Specific Features**: Integration with DifferentialEquations.jl and multiple dispatch
- [ ] **Cross-Language Compatibility**: Shared operation definitions between Python and Julia

### 10. Advanced Mathematical Features

#### Enhanced Type System
- [ ] **Generic Numeric Types**: Make symbolic optimizer generic over `T: NumericType`
- [ ] **Complex Numbers**: Support for complex-valued expressions
- [ ] **Matrix Operations**: Linear algebra primitives and operations

#### Advanced Compilation
- [ ] **LLVM Backend**: Direct LLVM IR generation for maximum performance
- [ ] **GPU Compilation**: CUDA/OpenCL code generation
- [ ] **Parallel Evaluation**: Multi-threaded expression evaluation

### 11. Performance and Production Features

#### Performance Optimization
- [ ] **SIMD Vectorization**: Leverage CPU vector instructions for bulk operations
- [ ] **Memory Pool Allocation**: Reduce allocation overhead in hot paths
- [ ] **Compilation Caching**: Cache compiled functions across sessions

#### Production Readiness
- [ ] **Comprehensive Benchmarking**: Performance regression testing and cross-language comparison
- [ ] **Documentation and Examples**: Complete API documentation and tutorial series
- [ ] **Error Handling**: Production-grade error handling, logging, and debugging support

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
- [x] **MCMC Integration Ready**: Direct compatibility with nuts-rs and other samplers
- [x] **Performance Optimization**: ~19M evaluations/second for compiled log-posterior functions
- [x] **Detailed Performance Profiling**: Stage-by-stage timing analysis with breakdown percentages
- [x] **Amortization Analysis**: Automatic calculation of compilation cost vs. runtime benefit
- [x] **dlopen2 Migration**: Replaced libloading with dlopen2 for better type safety and simplified architecture

### File Reorganization and Modularization (December 2024)
- [x] **Reorganized large files**: Split 2819-line `src/final_tagless.rs` into focused modules
- [x] **Modular structure**: Created `traits.rs`, `ast/`, `interpreters/`, `variables/` modules
- [x] **Comprehensive documentation**: Added examples and inline tests to all modules
- [x] **Backward compatibility**: All existing APIs continue to work unchanged
- [x] **Code quality**: 148/151 tests passing, clean compilation

### A-Normal Form (ANF) with Scope-Aware CSE (May 2025)
- [x] **ANF Intermediate Representation**: Complete transformation from `ASTRepr` to A-Normal Form
- [x] **Scope-Aware CSE**: Common subexpression elimination that respects variable lifetimes
- [x] **Hybrid Variable Management**: `VarRef::User(usize)` + `VarRef::Bound(u32)` system
- [x] **Clean Code Generation**: Produces readable, efficient Rust code
- [x] **Property-Based Testing**: Comprehensive test coverage including robustness testing

### Core Simplification Achievement (May 31, 2025)
- [x] **Statistical Computing Unification**: Proved statistical functions work perfectly with general system
- [x] **Fixed Core Methods**: `call_with_data()` now properly concatenates params and data
- [x] **Working Example**: `simplified_statistical_demo.rs` demonstrates Bayesian linear regression
- [x] **Architecture Validation**: Statistical computing via `f(β₀, β₁, x₁, y₁, x₂, y₂, ...)` pattern
- [x] **Performance Verified**: ~19M evaluations/second with the simplified approach
- [x] **Key Insight**: Statistical functionality is a special case of general mathematical expressions

### Legacy Code Cleanup (May 31, 2025)
- [x] **Deprecated Broken Methods**: Removed specialized statistical methods that ignored data parameters
- [x] **Updated Examples**: Created `simplified_statistical_demo.rs` using the general `call_multi_vars()` approach
- [x] **Cleaned Up Types**: Removed unnecessary statistical types (`RuntimeDataSpec`, `DataBinding`, `DataElementType`, `RuntimeSignature`, etc.)
- [x] **API Simplification**: Streamlined API surface to focus on the working general system
- [x] **Code Reduction**: Removed ~300 lines of unused statistical specialization code

### Domain Safety Improvements (May 31, 2025)
- [x] **Fixed ln(a/b) Rule**: Corrected domain safety issue where `ln(a/b) = ln(a) - ln(b)` was incorrectly applied to variables that could be negative
- [x] **Conservative Rule Application**: Made logarithm rules more conservative, only applying transformations when domain safety can be guaranteed
- [x] **Proptest Integration**: Used property-based testing to catch domain safety issues automatically
- [x] **NaN Prevention**: Eliminated NaN results from invalid mathematical transformations in symbolic optimization

## 🎯 **Next Priority: Advanced Domain-Aware Optimization**

Based on our successful native egglog integration and research into egglog's capabilities, the next step is to implement advanced domain-aware optimization using egglog's native abstract interpretation features.

### Key Discovery: egglog's Native Capabilities
- **Lattice-based Analysis**: egglog supports lattice semantics natively
- **Multiple Interacting Analyses**: Unlike egg (single analysis), egglog supports composable analyses  
- **Interval Analysis**: Proven in Herbie case study with domain-aware rules
- **Conditional Rules**: Rules can be gated on analysis results (e.g., `ln(a/b)` only if `a,b > 0`)

### Concrete Implementation Plan
Following the [Herbie/egglog paper](https://effect.systems/doc/egraphs-2023-egglog/paper.pdf):

1. **Interval Domain Implementation** (Week 1)
   - Add `Interval` datatype with proper merge functions
   - Implement interval arithmetic rules for basic operations
   - Add domain predicates (`ival-positive`, `ival-nonzero`)

2. **Domain-Aware Rules** (Week 2)  
   - Convert transcendental rules to use interval guards
   - Implement `ln(a/b) = ln(a) - ln(b)` with domain safety
   - Add `exp(ln(x)) = x` with positivity checks

3. **Advanced Analysis** (Week 3)
   - Multiple lattice analyses (intervals + not-equals)
   - Compositional analysis propagation
   - Cost-based extraction with domain information

4. **Integration & Testing** (Week 4)
   - Replace manual domain checks with native egglog analysis
   - Comprehensive test suite with domain edge cases
   - Performance benchmarking vs current approach

**Expected Outcome**: Domain-safe symbolic optimization that automatically prevents mathematical errors like the `ln(a/b)` issue we manually fixed, while enabling more aggressive optimizations when domain safety can be proven.

## 🔄 Current Status (May 31, 2025)

The library has reached a major milestone with the core simplification insight. The general mathematical expression system handles all use cases, including statistical computing, through the unified `call_multi_vars()` approach. This eliminates the need for complex specialized statistical infrastructure while maintaining full functionality.

**Key Achievements**: 
1. **Statistical functions** are now just mathematical expressions with more variables, using the pattern `f(β₀, β₁, x₁, y₁, x₂, y₂, ...)` instead of `f(params=[β₀, β₁], data=[x₁, y₁, x₂, y₂, ...])`
2. **Basic normalization** is complete, providing canonical form transformations that simplify the optimization pipeline and reduce egglog rule complexity by ~40%
3. **Rule system organization** is complete, with dynamic rule loading, multiple optimizer configurations, and clean separation of rule categories
4. **Domain safety improvements** including fixes for logarithm rules that were causing NaN results in edge cases
5. **Native egglog integration** is implemented, providing a foundation for future domain-aware optimization using egglog's native abstract interpretation capabilities

**Next Priority**: Implement domain-aware optimization using egglog's native interval analysis and conditional rewrite rules, following the approach demonstrated in the Herbie case study.

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

*Last updated: May 31, 2025*
*Status: Core simplification completed, ready for systematic implementation*
